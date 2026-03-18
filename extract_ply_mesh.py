import os, sys
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)
import torch
import numpy as np
import open3d as o3d
from argparse import ArgumentParser
import copy

from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import render_initial as render 
from scene import Scene, GaussianModel
# from utils.mesh_utils import GaussianExtractor
from datetime import datetime

def get_current_timestamp():
    return datetime.now().strftime("%d%m%y_%H%M%S")

# def post_process_mesh(mesh, cluster_to_keep=1000):
#     """
#     Post-process a mesh to filter out floaters and disconnected parts
#     """
#     import copy
#     mesh_0 = copy.deepcopy(mesh)
#     with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#         triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())
    
#     triangle_clusters = np.asarray(triangle_clusters)
#     cluster_n_triangles = np.asarray(cluster_n_triangles)
#     cluster_area = np.asarray(cluster_area)
    
#     # --- FIX: Ensure we don't request more clusters than actually exist ---
#     actual_clusters = len(cluster_n_triangles)
#     print(f"Post processing: Found {actual_clusters} clusters. Filtering...")
    
#     if actual_clusters == 0:
#         print("Warning: Mesh has no triangles!")
#         return mesh_0
        
#     safe_cluster_to_keep = min(cluster_to_keep, actual_clusters)
    
#     # Find the threshold for the n-th largest cluster
#     n_cluster = np.sort(cluster_n_triangles.copy())[-safe_cluster_to_keep]
#     n_cluster = max(n_cluster, 50) # filter meshes smaller than 50 triangles
    
#     triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
#     mesh_0.remove_triangles_by_mask(triangles_to_remove)
#     mesh_0.remove_unreferenced_vertices()
#     mesh_0.remove_degenerate_triangles()
    
#     print(f"Num vertices raw: {len(mesh.vertices)}")
#     print(f"Num vertices post: {len(mesh_0.vertices)}")
#     return mesh_0

def extract_ply(dataset: ModelParams, iteration: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        
        # Loading the scene automatically populates 'gaussians' with the checkpoint data
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        # Define the output directory and filename
        export_dir = os.path.join(dataset.model_path, "exported_models")
        os.makedirs(export_dir, exist_ok=True)
        
        output_path = os.path.join(export_dir, f"gaussians_iter_{scene.loaded_iter}_{get_current_timestamp()}.ply")
        
        # Export the Gaussians using the standard 3DGS save method
        print(f"Exporting 3D Gaussians to: {output_path}...")
        gaussians.save_ply(output_path)
        print("Extraction complete. This .ply file is ready for standard 3DGS viewers.")

# def extract_mesh(dataset: ModelParams, pipe: PipelineParams, args):
#     # Initialize Gaussians
#     print(f"Loading model from {dataset.model_path}, iteration {args.iteration}")
#     gaussians = GaussianModel(dataset.sh_degree)
#     print("Loaded Gaussian Model")
#     scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
#     print("Loaded Gaussian Scene")

#     bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    
#     # Initialize the Extractor
#     gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color) 
    
#     # 1. Reconstruct radiance fields (generate depth/rgb maps for the training cameras)
#     print("Reconstructing radiance fields from cameras...")
#     train_cameras = scene.getTrainCameras()
#     gaussExtractor.reconstruction(train_cameras)
    
#     # 2. Extract the mesh based on bounded/unbounded parameters
#     is_unbounded = 'ref_real' in dataset.source_path or args.unbounded
    
#     if args.marching_tetra_override:
#         print(f"Extracting mesh using marching tetrahedra algorithm")
#         mesh = gaussExtractor.extract_mesh_marching_tetrahedra_binary(texture_mesh=args.texture_mesh)
#     else:
#         if is_unbounded:
#             print(f"Extracting unbounded mesh at resolution {args.mesh_res}...")
#             mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
#         else:
#             print("Extracting bounded mesh using TSDF volume integration...")
#             depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0 else args.depth_trunc
#             voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
#             sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            
#             mesh = gaussExtractor.extract_mesh_bounded(
#                 voxel_size=voxel_size, 
#                 sdf_trunc=sdf_trunc, 
#                 depth_trunc=depth_trunc
#             )
        
#     # Post-process to remove floaters
#     if not args.marching_tetra_override:
#         mesh = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    
#     # Save the final mesh
#     output_dir = os.path.join(dataset.model_path, "exported_meshes")
#     os.makedirs(output_dir, exist_ok=True)
    
#     ply_path = os.path.join(output_dir, f'mesh_iter_{args.iteration:06d}_{get_current_timestamp()}.ply')
#     print(f"Saving mesh to {ply_path}")
#     # o3d.io.write_triangle_mesh(ply_path, mesh)
#     if args.marching_tetra_override:
#         # 'mesh' is a trimesh.Trimesh object here
#         mesh.export(ply_path)
#     else:
#         # 'mesh' is an open3d.geometry.TriangleMesh object here
#         o3d.io.write_triangle_mesh(ply_path, mesh)
#     print("Extraction complete!")

if __name__ == "__main__":
    # Set up command line arguments
    parser = ArgumentParser(description="Extract mesh from trained ref-gaussian model")
    
    # Add model and pipeline parameters
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    # Mesh extraction specific parameters
    parser.add_argument("--iteration", default=-1, type=int, help="Iteration to load. -1 loads the latest.")
    parser.add_argument("--mesh_res", default=1024, type=int, help="Resolution for unbounded grid or bounded voxel division")
    parser.add_argument("--unbounded", action="store_true", help="Force unbounded extraction even if 'ref_real' is not in path")
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help="Maximum depth range (calculated automatically if -1)")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help="Voxel size for TSDF (calculated automatically if -1)")
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help="Truncation value for TSDF (calculated automatically if -1)")
    parser.add_argument("--num_cluster", default=1000, type=int, help="Number of connected clusters to keep during post-processing")
    parser.add_argument("--texture_mesh", action="store_true", help="Enables texturing of the mesh for marching tetrahedral mode")
    parser.add_argument("--marching_tetra_override", action="store_true", help="Overrides others to use marching tetrahedral mesh extraction")
    parser.add_argument("--gaussian_ply_only", action="store_true", help="Overrides others to extract the raw gaussian ply only")
    
    args = get_combined_args(parser)
    
    if args.gaussian_ply_only:
        with torch.no_grad():
            extract_ply(model.extract(args), args.iteration) 
    else:
        with torch.no_grad():
            # extract_mesh(model.extract(args), pipeline.extract(args), args)
            print("Gaussian Mesh extraction is a work in progress. Please check back later!")
            pass