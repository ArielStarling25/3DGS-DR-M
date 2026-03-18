import os
import sys
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)
import torch
import random
import numpy as np
import trimesh
from tqdm import tqdm
from argparse import ArgumentParser
from datetime import datetime
from os import makedirs

# Standard 3DGS / 3DGS-DR Imports
from scene import Scene
from gaussian_renderer import render # Standard 2D render
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args

# Tetranerf / Mesh Utilities
from tetranerf.utils.extension import cpp
from utils.tetmesh import marching_tetrahedra

def get_tetra_points_fallback(gaussians, near=0.02, far=1e6, multiplier=2):
    """
    Fallback for GOF's gaussians.get_tetra_points()
    Generates a dense point cloud around the Gaussian centers for tetrahedralization
    """
    xyz = gaussians.get_xyz
    
    # Generate random perturbations around existing gaussians to create volume
    scales = gaussians.get_scaling
    max_scales = scales.max(dim=1)[0]
    
    points_list = [xyz]
    for _ in range(multiplier):
        noise = torch.randn_like(xyz) * max_scales.unsqueeze(1)
        points_list.append(xyz + noise)
        
    tetra_points = torch.cat(points_list, dim=0)
    
    # Approximate scale per point for the marching process
    points_scale = torch.cat([max_scales] * (multiplier + 1), dim=0)
    
    return tetra_points, points_scale

def evaluate_alpha_3dgs(points, views, gaussians, pipeline, background, kernel_size=None, return_color=False):
    """
    Evaluates alpha at 3D points. 
    Memory-safe PyTorch version with Nearest-Neighbor Color Approximation.
    """
    print("Warning: Using PyTorch approximation for 3D Alpha. Port `integrate` CUDA kernel for speed.")
    
    chunk_size = 1000 
    gaussian_chunk_size = 100000 
    
    final_alpha = torch.zeros((points.shape[0]), dtype=torch.float32, device="cuda")
    
    # Track the minimum distance to assign the color of the nearest Gaussian
    if return_color:
        final_color = torch.zeros((points.shape[0], 3), dtype=torch.float32, device="cuda")
        min_dist_sq_global = torch.full((points.shape[0],), float('inf'), device="cuda")
        
        # Get base colors from Spherical Harmonics (DC component)
        features = gaussians.get_features
        if len(features.shape) == 3:
            features_dc = features[:, 0, :] # Extract DC component
        else:
            features_dc = features
            
        # Standard conversion from SH degree 0 to RGB
        SH_C0 = 0.28209479177387814
        base_colors = torch.clamp(features_dc * SH_C0 + 0.5, 0.0, 1.0)

    means3D = gaussians.get_xyz
    opacities = gaussians.get_opacity
    cov3D_precomp = gaussians.get_covariance() 
    
    with torch.no_grad():
        for i in tqdm(range(0, points.shape[0], chunk_size), desc="Evaluating 3D Alpha"):
            pts_chunk = points[i:i+chunk_size]
            alpha_chunk = torch.zeros((pts_chunk.shape[0]), dtype=torch.float32, device="cuda")
            
            for j in range(0, means3D.shape[0], gaussian_chunk_size):
                means_chunk = means3D[j:j+gaussian_chunk_size]
                cov_chunk = cov3D_precomp[j:j+gaussian_chunk_size, 0] 
                opacities_chunk = opacities[j:j+gaussian_chunk_size]
                
                dist_sq = torch.cdist(pts_chunk, means_chunk, p=2)**2
                weights = torch.exp(-0.5 * dist_sq / (cov_chunk.unsqueeze(0) + 1e-6))
                alpha_chunk += (weights * opacities_chunk.T).sum(dim=1)
                
                if return_color:
                    # Find the nearest Gaussian in THIS chunk
                    min_dist_chunk, min_idx_chunk = torch.min(dist_sq, dim=1)
                    
                    # Update colors if we found a closer Gaussian than before
                    current_min_dists = min_dist_sq_global[i:i+chunk_size]
                    update_mask = min_dist_chunk < current_min_dists
                    
                    # Apply updates
                    min_dist_sq_global[i:i+chunk_size][update_mask] = min_dist_chunk[update_mask]
                    global_idx = j + min_idx_chunk
                    final_color[i:i+chunk_size][update_mask] = base_colors[global_idx][update_mask]
                
            final_alpha[i:i+chunk_size] = torch.clamp(alpha_chunk, max=1.0)

    alpha = 1.0 - final_alpha 
    
    if return_color:
        return alpha, final_color
    return alpha

def get_current_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

@torch.no_grad()
def marching_tetrahedra_with_binary_search(model_path, name, iteration, views, gaussians, pipeline, background, filter_mesh: bool, texture_mesh: bool, near: float, far: float):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "fusion")
    makedirs(render_path, exist_ok=True)
    
    # 1. Get Tetra Points (Using fallback if CUDA method is missing)
    if hasattr(gaussians, 'get_tetra_points'):
        points, points_scale = gaussians.get_tetra_points(views, near, far)
    else:
        points, points_scale = get_tetra_points_fallback(gaussians, near, far)

    cells_path = os.path.join(render_path, "cells.pt")
    cells = None

    if os.path.exists(cells_path):
        print("Found existing cells.pt, checking validity...")
        loaded_cells = torch.load(cells_path)
        if loaded_cells.max() < points.shape[0]:
            print("Cells are valid. Loading...")
            cells = loaded_cells
        else:
            print(f"Cache mismatch. Ignoring cache.")

    if cells is None:
        print("Triangulating points to create cells...")
        cells = cpp.triangulate(points.cpu().contiguous()).cuda()
        torch.save(cells, cells_path)
        print("Cells saved.")
    
    # 2. Evaluate Alpha
    alpha = evaluate_alpha_3dgs(points, views, gaussians, pipeline, background)

    vertices = points.cuda()[None]
    tets = cells.cuda().long()

    def alpha_to_sdf(alpha):    
        return (alpha - 0.5)[None]
    
    sdf = alpha_to_sdf(alpha)
    
    torch.cuda.empty_cache()
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(vertices, tets, sdf, points_scale[None])
    torch.cuda.empty_cache()
    
    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]
    faces = faces_list[0].cpu().numpy()
    
    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale
    
    n_binary_steps = 8
    for step in range(n_binary_steps):
        print(f"Binary search in step {step}")
        mid_points = (left_points + right_points) / 2
        
        alpha = evaluate_alpha_3dgs(mid_points, views, gaussians, pipeline, background)
        mid_sdf = alpha_to_sdf(alpha).squeeze().unsqueeze(-1)
        
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
    
        points = (left_points + right_points) / 2
        if step != 7:
            continue
        
        if texture_mesh:
            _, color = evaluate_alpha_3dgs(points, views, gaussians, pipeline, background, return_color=True)
            vertex_colors = (torch.clamp(color, 0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
        else:
            vertex_colors = None
            
        mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, vertex_colors=vertex_colors, process=False)
        
        if filter_mesh:
            mask = (distance <= scale).cpu().numpy()
            face_mask = mask[faces].all(axis=1)
            mesh.update_vertices(mask)
            mesh.update_faces(face_mask)
        
        mesh.export(os.path.join(render_path, f"mesh_binary_search_{step}_{get_current_timestamp()}.ply"))

def extract_mesh(dataset: ModelParams, iteration: int, pipeline: PipelineParams, filter_mesh: bool, texture_mesh: bool, near: float, far: float):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        cams = scene.getTrainCameras()
        print("Starting Marching Tetrahedra + Binary Search for 3DGS-DR")
        marching_tetrahedra_with_binary_search(dataset.model_path, "test", iteration, cams, gaussians, pipeline, background, filter_mesh, texture_mesh, near, far)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--filter_mesh", action="store_true")
    parser.add_argument("--texture_mesh", action="store_true")
    parser.add_argument("--near", default=0.02, type=float)
    parser.add_argument("--far", default=1e6, type=float)
    
    args = get_combined_args(parser)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    extract_mesh(model.extract(args), args.iteration, pipeline.extract(args), args.filter_mesh, args.texture_mesh, args.near, args.far)