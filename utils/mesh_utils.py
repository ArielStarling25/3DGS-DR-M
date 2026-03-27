#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
import open3d as o3d
import trimesh

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())
    
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

class SimpleCamParams:
    def __init__(self, intrinsic, extrinsic):
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=int(viewpoint_cam.image_width),
            height=int(viewpoint_cam.image_height),
            cx = float(intrins[0,2].item()),
            cy = float(intrins[1,2].item()), 
            fx = float(intrins[0,0].item()), 
            fy = float(intrins[1,1].item())
        )

        extrinsic = np.ascontiguousarray((viewpoint_cam.world_view_transform.T).cpu().numpy(), dtype=np.float64)
        camera = SimpleCamParams(intrinsic, extrinsic)
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        # self.alphamaps = []
        self.rgbmaps = []
        # self.normals = []
        # self.depth_normals = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            rgb = render_pkg['render']
            depth = render_pkg['depth_map']
            normal = torch.nn.functional.normalize(render_pkg['normal_map'], dim=0)
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            # depth = render_pkg['surf_depth']
            # depth_normal = render_pkg['surf_normal']
            # self.rgbmaps.append(rgb.cpu())
            # self.depthmaps.append(depth.cpu())
            # self.alphamaps.append(alpha.cpu())
            # self.normals.append(normal.cpu())
            # self.depth_normals.append(depth_normal.cpu())
        
        # self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        # self.depthmaps = torch.stack(self.depthmaps, dim=0)
        # self.alphamaps = torch.stack(self.alphamaps, dim=0)
        # self.depth_normals = torch.stack(self.depth_normals, dim=0)
        self.estimate_bounding_sphere()

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        import math
        import skimage.measure
        print("Extracting Bounded Mesh using PURE PyTorch TSDF & skimage (Bypassing Open3D Core Dump)...")
        
        R = float(self.radius)
        N = int((R * 2.0) / voxel_size)
        
        # Memory Limit Mitigation: Cap resolution aggressively so PyTorch doesn't cause OOM!
        if N > 512:
            N = 512
            voxel_size = (R * 2.0) / N
            print(f"WARNING: Initial voxel grid too large! Capped resolution at {N}^3 (voxel size scaled to {voxel_size:.4f})")
            
        print(f"Allocating {N}x{N}x{N} TSDF Volume natively on GPU...")
        
        grid_x = torch.linspace(-R, R, N, device="cuda")
        grid_y = torch.linspace(-R, R, N, device="cuda")
        grid_z = torch.linspace(-R, R, N, device="cuda")
        X, Y, Z = torch.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
        
        pts_world = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3) + self.center
        
        # Initialize TSDF buffers directly on CUDA tensor logic
        tsdf = torch.ones(pts_world.shape[0], device="cuda")
        weights = torch.zeros(pts_world.shape[0], device="cuda")
        rgb_grid = torch.zeros((pts_world.shape[0], 3), device="cuda")
        
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="Integrating frames via PyTorch"):
            depthmap = self.depthmaps[i].cuda()  # [1, H, W]
            rgbmap = self.rgbmaps[i].cuda()      # [3, H, W]

            if mask_backgrond and getattr(viewpoint_cam, 'gt_alpha_mask', None) is not None:
                mask = viewpoint_cam.gt_alpha_mask.cuda() < 0.5
                depthmap[mask] = 0.0

            H = viewpoint_cam.image_height
            W = viewpoint_cam.image_width
            fx = W / (2.0 * math.tan(viewpoint_cam.FoVx / 2.0))
            fy = H / (2.0 * math.tan(viewpoint_cam.FoVy / 2.0))
            cx = W / 2.0
            cy = H / 2.0

            w2c = viewpoint_cam.world_view_transform.T.cuda() 
            pts_cam = torch.cat([pts_world, torch.ones_like(pts_world[:, :1])], dim=-1) @ w2c 
            
            x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
            
            # Prune to memory arrays matching validity metrics
            valid_z = (z > 0.0) & (z < depth_trunc)
            if not valid_z.any(): continue
            
            u = (x[valid_z] * fx / z[valid_z]) + cx
            v = (y[valid_z] * fy / z[valid_z]) + cy
            
            valid_proj = (u >= 0) & (u < W-1) & (v >= 0) & (v < H-1)
            if not valid_proj.any(): continue
            
            mapper = valid_z.nonzero().squeeze()[valid_proj]
            
            u_valid = u[valid_proj]
            v_valid = v[valid_proj]
            z_valid = z[valid_z][valid_proj]
            
            # Sub-Pixel array normalization for Grid Sampling constraints
            u_ndc = (u_valid / (W - 1)) * 2.0 - 1.0
            v_ndc = (v_valid / (H - 1)) * 2.0 - 1.0
            
            uv_ndc = torch.stack([u_ndc, v_ndc], dim=-1).unsqueeze(0).unsqueeze(0) 
            
            sampled_depth = torch.nn.functional.grid_sample(depthmap.unsqueeze(0), uv_ndc, align_corners=True, padding_mode='zeros').squeeze()
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.unsqueeze(0), uv_ndc, align_corners=True, padding_mode='zeros').squeeze().T
            
            valid_depth = sampled_depth > 0.0
            surface_dist = sampled_depth - z_valid
            
            # Apply strict SDF logical bindings
            mask_final = valid_depth & (surface_dist > -sdf_trunc)
            
            final_mapper = mapper[mask_final]
            z_dist = surface_dist[mask_final]
            sdf = torch.clamp(z_dist / sdf_trunc, min=-1.0, max=1.0)
            
            old_w = weights[final_mapper]
            new_w = old_w + 1.0
            
            # Weighted average accumulation to arrays
            tsdf[final_mapper] = (tsdf[final_mapper] * old_w + sdf) / new_w
            rgb_grid[final_mapper] = (rgb_grid[final_mapper] * old_w.unsqueeze(-1) + sampled_rgb[mask_final]) / new_w.unsqueeze(-1)
            weights[final_mapper] = new_w
            
        print("Running skimage marching cubes asynchronously...")
        tsdf_vol = tsdf.reshape(N, N, N).cpu().numpy()
        
        if tsdf_vol.max() < 0.0 or tsdf_vol.min() > 0.0:
            print("WARNING: TSDF entirely uniform/empty!")
            verts, faces = np.zeros((0,3)), np.zeros((0,3))
        else:
            verts, faces, normals, values = skimage.measure.marching_cubes(tsdf_vol, level=0.0, spacing=(voxel_size, voxel_size, voxel_size))
            
        verts = verts - R + self.center.cpu().numpy()
        
        print(f"Marching cubes completed! Polygons: {len(faces)}. Building lightweight Trimesh wrapper...")
        import trimesh
        
        if len(verts) > 0:
            v_idx = ((verts - self.center.cpu().numpy() + R) / voxel_size).astype(int)
            v_idx = np.clip(v_idx, 0, N-1)
            colors = rgb_grid.reshape(N,N,N,3).cpu().numpy()[v_idx[:,0], v_idx[:,1], v_idx[:,2]]
            
            # Map [0, 1] TSDF colors to native Trimesh uint8 [0, 255] arrays.
            colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors, process=False)
        else:
            mesh = trimesh.Trimesh()
        
        print("Mesh extracted securely via native PyTorch Engine!")
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        
        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # update weight
                weights[mask_proj] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )
        
        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        vis_path = os.path.join(path, "vis")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            # save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
            # save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))
