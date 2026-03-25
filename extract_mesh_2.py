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

# 3DGS-DR Imports
from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args

# Tetranerf / Mesh Utilities
from tetranerf.utils.extension import cpp
from trimesh.smoothing import filter_laplacian
from utils.tetmesh import marching_tetrahedra

def get_current_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

@torch.no_grad()
def extract_mesh_marching_tetrahedra_binary(model_path, name, iteration, gaussians, filter_mesh=True, texture_mesh=True, n_binary_steps=8):
    """
    Adapted from Own Ref-Gaussian ver and Gaussian Opacity Fields implementation
    Uses full inverse covariance matrices and bounding grid for accurate evaluation
    """
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "fusion")
    makedirs(render_path, exist_ok=True)
    
    print("Starting Marching Tetrahedra with Binary Search extraction...")

    # Generate Base Points for Tetrahedralization
    gaussian_points = gaussians.get_xyz.detach()
    
    # Calculate scene bounds for the grid
    center = gaussian_points.mean(dim=0)
    radius = (gaussian_points - center).norm(dim=-1).max().item() * 1.2 
    
    # Create a sparse background grid based on the estimated bounding sphere
    grid_res = 32 # 32, 64, 128, 254 depending on how much RAM you have
    x = torch.linspace(-radius, radius, grid_res, device="cuda")
    y, z = x.clone(), x.clone()
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3) + center
    
    points = torch.cat([gaussian_points, grid_points], dim=0)
    
    # gaussian_scales = torch.exp(gaussians.get_scaling).max(dim=-1)[0]
    gaussian_scales = gaussians.get_scaling.detach().max(dim=-1)[0]
    grid_scales = torch.ones(grid_points.shape[0], device="cuda") * (radius * 2 / grid_res)
    points_scale = torch.cat([gaussian_scales, grid_scales], dim=0)

    # Triangulate Points into Tetrahedra Cells
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
        print("Triangulating points...")
        cells = cpp.triangulate(points.cpu().contiguous()).cuda()
        torch.save(cells, cells_path)
        print("Cells saved.")

    print("Preparing Gaussian Covariances and Colors for Volumetric Evaluation...")
    
    xyz = gaussians.get_xyz.detach()
    opacity = gaussians.get_opacity.detach().flatten()
    
    # Filter out visually insignificant Gaussians to speed up math
    mask = opacity > 0.05
    xyz = xyz[mask]
    opacity = opacity[mask]
    scales = gaussians.get_scaling.detach()[mask]
    rotations = gaussians.get_rotation.detach()[mask]
    
    # Extract base colors (Spherical Harmonics DC component)
    features = gaussians.get_features.detach()
    if len(features.shape) == 3:
        sh_dc = features[mask, 0, :]
    else:
        sh_dc = features[mask]
        
    SH2RGB = 0.28209479177387814
    base_colors = torch.clamp(sh_dc * SH2RGB + 0.5, 0.0, 1.0) 

    if scales.shape[1] == 2:
        thickness = torch.ones((scales.shape[0], 1), device="cuda") * 0.01
        scales_3d = torch.cat([scales, thickness], dim=1)
    else:
        scales_3d = scales

    # Compute Inverse Covariance Matrices (Sigma^-1)
    r = rotations[:, 0]
    x = rotations[:, 1]
    y = rotations[:, 2]
    z = rotations[:, 3]
    
    R = torch.zeros((xyz.shape[0], 3, 3), device="cuda")
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    
    inv_S = torch.zeros((xyz.shape[0], 3, 3), device="cuda")
    inv_S[:, 0, 0] = 1.0 / (scales_3d[:, 0] ** 2 + 1e-7)
    inv_S[:, 1, 1] = 1.0 / (scales_3d[:, 1] ** 2 + 1e-7)
    inv_S[:, 2, 2] = 1.0 / (scales_3d[:, 2] ** 2 + 1e-7)
    
    inv_cov = torch.bmm(torch.bmm(R, inv_S), R.transpose(1, 2))

    def evaluate_sdf_and_color(eval_points):
        """
        Evaluates the 3D volumetric density and weighted color of the Gaussians
        """
        density_threshold = 0.5 
        num_points = eval_points.shape[0]
        M_gaussians = xyz.shape[0]
        
        total_density = torch.zeros(num_points, device="cuda")
        total_colors = torch.zeros((num_points, 3), device="cuda")
        
        # chunk_size_pts = 1000  
        # chunk_size_gaussians = 50000 
        chunk_size_pts = 4000  
        chunk_size_gaussians = 100000 
        
        # --- PRECOMPUTATION PHASE ---
        # A = inv_cov [M, 3, 3], mu = xyz [M, 3]
        
        # Term 3: mu^T A mu  -> Shape: [M]
        mu_A_mu = torch.einsum('mi,mij,mj->m', xyz, inv_cov, xyz)
        
        # Term 2 partial: A * mu -> Shape: [M, 3]
        V = torch.einsum('mij,mj->mi', inv_cov, xyz)
        
        # Term 1 partial: Flattened A -> Shape: [M, 9]
        A_flat = inv_cov.reshape(M_gaussians, 9)
        # ----------------------------
        
        for i in tqdm(range(0, num_points, chunk_size_pts), desc="Evaluating SDF", leave=False):
            end_pts = min(i + chunk_size_pts, num_points)
            pts_chunk = eval_points[i:end_pts] 
            
            density_accumulator = torch.zeros(pts_chunk.shape[0], device="cuda")
            color_accumulator = torch.zeros((pts_chunk.shape[0], 3), device="cuda")
            
            # --- TERM 1 PREP ---
            # Precompute flattened x*x^T for the current points chunk -> Shape: [N_chunk, 9]
            X_xx = torch.einsum('ni,nj->nij', pts_chunk, pts_chunk)
            X_flat = X_xx.reshape(pts_chunk.shape[0], 9)
            
            for j in range(0, M_gaussians, chunk_size_gaussians):
                end_g = min(j + chunk_size_gaussians, M_gaussians)
                
                opacity_g = opacity[j:end_g]        
                colors_g = base_colors[j:end_g]      
                
                # Slice precomputed variables
                mu_A_mu_g = mu_A_mu[j:end_g]
                V_g = V[j:end_g]
                A_flat_g = A_flat[j:end_g]
                
                # --- PURE MATRIX MULTIPLICATION (cuBLAS) ---
                # 1. x^T A x  -> Shape: [N_chunk, M_chunk]
                term1 = torch.matmul(X_flat, A_flat_g.T) 
                
                # 2. -2 x^T A mu -> Shape: [N_chunk, M_chunk]
                term2 = -2.0 * torch.matmul(pts_chunk, V_g.T)
                
                # 3. Sum terms. (Broadcasting handles adding the [M_chunk] array)
                # We use clamp(min=0.0) to prevent floating-point inaccuracies from causing tiny negative distances
                dist_sq = torch.clamp(term1 + term2 + mu_A_mu_g, min=0.0)
                
                # --- ACCUMULATION ---
                # Broadcasting applies the [M_chunk] opacity across the [N_chunk, M_chunk] tensor
                gauss_density = opacity_g * torch.exp(-0.5 * dist_sq)
                
                density_accumulator += torch.sum(gauss_density, dim=1)
                color_accumulator += torch.matmul(gauss_density, colors_g)
            
            total_density[i:end_pts] = density_accumulator
            total_colors[i:end_pts] = color_accumulator / (density_accumulator.unsqueeze(-1) + 1e-7)
            
        sdf = density_threshold - total_density 
        return sdf.unsqueeze(-1), total_colors
    
    # def evaluate_sdf_and_color(eval_points):
    #     """
    #     Evaluates the 3D volumetric density and weighted color of the Gaussians.
    #     """
    #     density_threshold = 0.5 
    #     num_points = eval_points.shape[0]
    #     M_gaussians = xyz.shape[0]
        
    #     total_density = torch.zeros(num_points, device="cuda")
    #     total_colors = torch.zeros((num_points, 3), device="cuda")
        
    #     chunk_size_pts = 1000  
    #     chunk_size_gaussians = 50000 
        
    #     for i in tqdm(range(0, num_points, chunk_size_pts), desc="Evaluating SDF", leave=False):
    #         end_pts = min(i + chunk_size_pts, num_points)
    #         pts_chunk = eval_points[i:end_pts] 
            
    #         density_accumulator = torch.zeros(pts_chunk.shape[0], device="cuda")
    #         color_accumulator = torch.zeros((pts_chunk.shape[0], 3), device="cuda")
            
    #         for j in range(0, M_gaussians, chunk_size_gaussians):
    #             end_g = min(j + chunk_size_gaussians, M_gaussians)
                
    #             xyz_g = xyz[j:end_g]                 
    #             inv_cov_g = inv_cov[j:end_g]         
    #             opacity_g = opacity[j:end_g]         
    #             colors_g = base_colors[j:end_g]      
                
    #             delta = pts_chunk.unsqueeze(1) - xyz_g.unsqueeze(0) 
    #             left = torch.matmul(delta.unsqueeze(-2), inv_cov_g.unsqueeze(0)) 
    #             dist_sq = torch.matmul(left, delta.unsqueeze(-1)).squeeze(-1).squeeze(-1) 
                
    #             gauss_density = opacity_g.unsqueeze(0) * torch.exp(-0.5 * dist_sq)
                
    #             density_accumulator += torch.sum(gauss_density, dim=1)
    #             color_accumulator += torch.matmul(gauss_density, colors_g)
            
    #         total_density[i:end_pts] = density_accumulator
    #         total_colors[i:end_pts] = color_accumulator / (density_accumulator.unsqueeze(-1) + 1e-7)
            
    #         torch.cuda.empty_cache()
        
    #     sdf = density_threshold - total_density 
    #     return sdf.unsqueeze(-1), total_colors

    # Initialise Marching Tetrahedra
    print("Evaluating initial SDF...")
    sdf, _ = evaluate_sdf_and_color(points)
    
    vertices = points.cuda()[None]
    tets = cells.cuda().long()
    
    torch.cuda.empty_cache()
    print("Starting Marching Tetrahedra")
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(vertices, tets, sdf[None], points_scale[None])
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
    scale_threshold = left_scale + right_scale

    # Binary Search Refinement
    for step in range(n_binary_steps):
        print(f"Binary search refinement step {step+1}/{n_binary_steps}")
        mid_points = (left_points + right_points) / 2.0
        mid_sdf, _ = evaluate_sdf_and_color(mid_points)
        
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))
        
        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]

    final_points = (left_points + right_points) / 2.0
    
    if texture_mesh:
        print("Extracting vertex colors...")
        _, final_colors = evaluate_sdf_and_color(final_points)
        vertex_colors = (final_colors.cpu().numpy() * 255).astype(np.uint8)
    else:
        vertex_colors = None

    mesh = trimesh.Trimesh(vertices=final_points.cpu().numpy(), faces=faces, vertex_colors=vertex_colors, process=False)

    if filter_mesh:
        print("Filtering mesh based on Gaussian scales...")
        mask = (distance <= scale_threshold).cpu().numpy()
        face_mask = mask[faces].all(axis=1)
        mesh.update_vertices(mask)
        mesh.update_faces(face_mask)

    print("Applying Laplacian smoothing...")
    # Adjust iterations (e.g., 3 to 5) depending on how smooth you want it
    mesh.fix_normals()
    # Smooth the mesh, but explicitly turn OFF volume preservation
    filter_laplacian(mesh, iterations=3, volume_constraint=False)

    out_file = os.path.join(render_path, f"mesh_binary_search_{get_current_timestamp()}.ply")
    mesh.export(out_file)
    print(f"Mesh successfully exported to {out_file}")

def extract_mesh(dataset: ModelParams, iteration: int, pipeline: PipelineParams, filter_mesh: bool, texture_mesh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        extract_mesh_marching_tetrahedra_binary(
            model_path=dataset.model_path, 
            name="test", 
            iteration=iteration, 
            gaussians=gaussians, 
            filter_mesh=filter_mesh, 
            texture_mesh=texture_mesh
        )

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--filter_mesh", action="store_true")
    parser.add_argument("--texture_mesh", action="store_true")
    
    args = get_combined_args(parser)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    extract_mesh(model.extract(args), args.iteration, pipeline.extract(args), args.filter_mesh, args.texture_mesh)