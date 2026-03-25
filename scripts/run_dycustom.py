import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
from datetime import datetime

# EDIT YOUR TRAINING LIST HERE
training_list = [
    # Nerf Synthetic
    # {"name": "lego", "group": "nerf_synthetic", "iteration": 50000, "debug_render": True, "add_params": "--white_background --densification_interval_when_prop 100"},
    # {"name": "drums", "group": "nerf_synthetic", "iteration": 50000, "debug_render": True, "add_params": "--white_background --densification_interval_when_prop 100"},
    # {"name": "ship", "group": "nerf_synthetic", "iteration": 50000, "debug_render": True, "add_params": "--white_background --densification_interval_when_prop 100"},
    # {"name": "hotdog", "group": "nerf_synthetic", "iteration": 50000, "debug_render": True, "add_params": "--white_background --densification_interval_when_prop 100"},
    # {"name": "ficus", "group": "nerf_synthetic", "iteration": 50000, "debug_render": True, "add_params": "--white_background --densification_interval_when_prop 100"},
    # {"name": "mic", "group": "nerf_synthetic", "iteration": 50000, "debug_render": True, "add_params": "--white_background --densification_interval_when_prop 100"},
    # {"name": "materials", "group": "nerf_synthetic", "iteration": 50000, "debug_render": True, "add_params": "--white_background --densification_interval_when_prop 100"},
    # # Ref Nerf
    # {"name": "ball", "group": "refnerf", "iteration": 50000, "debug_render": True, "add_params": "--white_background"},
    # {"name": "car", "group": "refnerf", "iteration": 50000, "debug_render": True, "add_params": "--white_background"},
    # {"name": "coffee", "group": "refnerf", "iteration": 50000, "debug_render": True, "add_params": "--white_background"},
    # Ref Real
    {"name": "sedan", "group": "ref_real", "factor": 2, "iteration": 61000, "add_iter": 36000, "add_params": "--longer_prop_iter 36_000 --use_env_scope --env_scope_center -0.032 0.808 0.751 --env_scope_radius 2.138"},
    {"name": "toycar", "group": "ref_real", "factor": 2, "iteration": 61000, "add_iter": 36000, "add_params": "--longer_prop_iter 36_000 --use_env_scope --env_scope_center 0.6810 0.8080 4.4550 --env_scope_radius 2.707"},
    {"name": "gardenspheres", "group": "ref_real", "factor": 2, "iteration": 61000, "add_iter": 36000, "add_params": "--longer_prop_iter 36_000 --use_env_scope --env_scope_center -0.2270 1.9700 1.7740 --env_scope_radius 0.974"},
    {"name": "sedan", "group": "ref_real", "factor": 2, "iteration": 61000},
    {"name": "toycar", "group": "ref_real", "factor": 2, "iteration": 61000},
    {"name": "gardenspheres", "group": "ref_real", "factor": 2, "iteration": 61000},
]

# Testing if the additional parameters really affect the overall quality, if so then thats an issue for scalability...

# training_list = [
#     {"name": "gardenspheres", "group": "ref_real", "factor": 4, "iteration": 61000}, 
#     {"name": "gardenspheres", "group": "ref_real", "factor": 4, "iteration": 61001, "add_params": "--longer_prop_iter 36_000 --use_env_scope --env_scope_center -0.2270 1.9700 1.7740 --env_scope_radius 0.974"},
#     {"name": "sedan", "group": "ref_real", "iteration": 61000, "add_params": "--longer_prop_iter 36_000 --use_env_scope --env_scope_center -0.032 0.808 0.751 --env_scope_radius 2.138"},
#     {"name": "toycar", "group": "ref_real", "iteration": 61000, "add_params": "--longer_prop_iter 36_000 --use_env_scope --env_scope_center 0.6810 0.8080 4.4550 --env_scope_radius 2.707"},
# ]

scenes = training_list
factors = [2] * len(scenes)

excluded_gpus = set([])

output_dir = "exp_Custom"

dry_run = False
RESULTS_ONLY = False
MESH_EXTRACT_ONLY = False
ENABLE_MASK = False

jobs = list(zip(scenes, factors))

def train_scene(gpu, scene, factor=None):
    current_file_path = Path(__file__).resolve()
    scripts_dir = current_file_path.parent
    project_root = scripts_dir.parent
    
    dataset_path = os.path.join(project_root, "data", scene['group'], scene['name'])
    output_directory = os.path.join(output_dir, scene['group'], scene['name'])
    set_iterations = scene['iteration']
    
    print("Dataset Path set to: ", dataset_path)
    print("Output Directory set to: ", output_directory)

    train_duration = 0.0
    mesh_duration = 0.0

    # Train
    cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 train.py -s {dataset_path} -m {output_directory} --eval --iterations {set_iterations}"
    if 'factor' in scene:
        if scene['factor'] > 0:
            cmd += (" " + f"--images images_{scene['factor']} --resolution {scene['factor']}")
    if 'add_params' in scene:
        cmd += (" " + scene['add_params'])
    print(cmd)
    start_train = time.perf_counter()
    if not dry_run:
        os.system(cmd)
    train_duration = time.perf_counter() - start_train

    # Handling addtional iterations from additional parameters
    if 'add_iter' in scene:
        if scene['add_iter'] > 0:
            set_iterations += scene['add_iter']

    # Extract Splat PLY
    cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 extract_ply_mesh.py --white_background --model_path {output_directory} --iteration {set_iterations} --gaussian_ply_only"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    # Extract Mesh PLY
    # cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 extract_mesh.py -m {output_directory} --iteration {set_iterations} --texture_mesh"
    cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 extract_mesh_2.py -m {output_directory} --iteration {set_iterations} --texture_mesh"
    print(cmd)
    start_mesh = time.perf_counter()
    if not dry_run:
        os.system(cmd)
    mesh_duration = time.perf_counter() - start_mesh
    
    # Evaluation + render (rgb, normal, mesh) images
    cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 eval.py --white_background --save_images --model_path {output_directory} --iteration {set_iterations}"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    # Debug renders
    if 'debug_render' in scene:
        if scene['debug_render']:
            cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 scripts/debug_renders.py --gt {os.path.join(dataset_path, 'test')} --renders {os.path.join(output_directory, 'test', f'ours_{set_iterations}', 'renders', 'rgb')} --out {output_directory}"
            print(cmd)
            if not dry_run:
                os.system(cmd)

    return {"train_time": train_duration, "mesh_time": mesh_duration}

def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene['name']}\n")
    timings = train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene['name']}\n")
    return timings

def dispatch_jobs(jobs, executor, excluded_gpus=None):
    if excluded_gpus is None:
        excluded_gpus = set()

    run_timestamp = datetime.now().strftime("%d%m%y_run_%H%M%S")
    
    future_to_job = {}
    reserved_gpus = set()
    
    completed_jobs_stats_per_scene = {} 
    total_start_perf = time.perf_counter()

    def report_stats(scene_name, current_stats, final=False):
        # Column widths
        w_name, w_dur, w_sub, w_time, w_m = 30, 10, 10, 20, 8
        
        # Dynamically set log dir based on the job's 'name'
        group_name = current_stats[0]['group'] if current_stats else "unknown"
        specific_log_dir = os.path.join(output_dir, group_name, scene_name, "run_logs")
        os.makedirs(specific_log_dir, exist_ok=True)
        log_file_path = os.path.join(specific_log_dir, f"{run_timestamp}.txt")

        # Headers for timings and metrics
        header = (f"{'Job Detail':<{w_name}} | {'Total(s)':<{w_dur}} | "
                  f"{'Train(s)':<{w_sub}} | {'Mesh(s)':<{w_sub}} | "
                  f"{'Start Time':<{w_time}} | {'End Time':<{w_time}} | "
                  f"{'PSNR':<{w_m}} | {'SSIM':<{w_m}} | {'LPIPS':<{w_m}} | {'FPS':<{w_m}}")
        
        separator = "-" * len(header)
        lines = ["\n" + separator, header, separator]
        
        for stat in current_stats:
            iteration = stat.get('iteration', 50000)
            
            results_path = os.path.join(output_dir, stat['group'], scene_name, 'metric.txt')
            
            # Helper function to extract CSV metrics: psnr:val,ssim:val,lpips:val,fps:val
            def get_metrics(path):
                if os.path.exists(path):
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            pairs = content.split(',')
                            m = {}
                            for p in pairs:
                                if ':' in p:
                                    k, v = p.split(':', 1)
                                    m[k.strip().lower()] = float(v.strip())
                            
                            return (f"{m.get('psnr', 0):.4f}", 
                                    f"{m.get('ssim', 0):.4f}", 
                                    f"{m.get('lpips', 0):.4f}", 
                                    f"{m.get('fps', 0):.2f}")
                    except Exception as e:
                        print(f"Error reading {path}: {e}")
                return "N/A", "N/A", "N/A", "N/A"

            psnr, ssim, lpips, fps = get_metrics(results_path)

            lines.append(f"{str(stat['name']):<{w_name}} | "
                         f"{stat['duration']:<{w_dur}.2f} | "
                         f"{stat.get('train_time', 0):<{w_sub}.2f} | "
                         f"{stat.get('mesh_time', 0):<{w_sub}.2f} | "
                         f"{stat['start']:<{w_time}} | "
                         f"{stat['end']:<{w_time}} | "
                         f"{psnr:<{w_m}} | {ssim:<{w_m}} | {lpips:<{w_m}} | {fps:<{w_m}}")
            
        lines.append(separator + "\n")
        report_text = "\n".join(lines)
        
        print(f"Logging stats for {scene_name} -> {log_file_path}")
        print(report_text)
        
        with open(log_file_path, "a", encoding="utf-8") as f:
            status = "FINAL SUMMARY" if final else "INTERMEDIATE UPDATE"
            f.write(f"\n[{status} - {datetime.now().strftime('%H:%M:%S')}]\n")
            f.write(report_text)
            if final:
                total_duration = time.perf_counter() - total_start_perf
                f.write(f"\nBATCH COMPLETE. Total Batch Duration: {total_duration:.2f} seconds.\n")

    while jobs or future_to_job:
        try:
            all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1, maxLoad=0.1))
        except Exception:
            all_available_gpus = set() 

        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)
        
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            
            start_perf = time.perf_counter()
            start_dt = datetime.now()
            
            future = executor.submit(worker, gpu, *job)
            
            future_to_job[future] = {
                "gpu": gpu, 
                "job": job, 
                "start_perf": start_perf,
                "start_dt": start_dt
            }
            reserved_gpus.add(gpu)

        done_futures = [future for future in future_to_job if future.done()]
        
        for future in done_futures:
            data = future_to_job.pop(future)
            gpu = data["gpu"]
            job = data["job"]
            scene = job[0]
            scene_name = scene['name'] 
            
            start_perf = data["start_perf"]
            start_dt = data["start_dt"]
            
            end_perf = time.perf_counter()
            end_dt = datetime.now()
            duration = end_perf - start_perf
            reserved_gpus.discard(gpu)
            
            train_time = 0.0
            mesh_time = 0.0
            
            try:
                job_results = future.result()
                if isinstance(job_results, dict):
                    train_time = job_results.get("train_time", 0.0)
                    mesh_time = job_results.get("mesh_time", 0.0)
            except Exception as exc:
                print(f"Job {job} generated an exception: {exc}")
            
            time_fmt = "%d/%m/%y | %H:%M:%S"
            
            if scene_name not in completed_jobs_stats_per_scene:
                completed_jobs_stats_per_scene[scene_name] = []

            job_display_name = f"{scene['group']}_{scene_name}_iter{scene['iteration']}"

            completed_jobs_stats_per_scene[scene_name].append({
                "name": job_display_name,
                "group": scene['group'],
                "duration": duration,
                "train_time": train_time,
                "mesh_time": mesh_time,
                "start": start_dt.strftime(time_fmt),
                "end": end_dt.strftime(time_fmt),
                "iteration": scene['iteration']    
            })
            
            print(f"Job finished - Releasing GPU {gpu}")
            report_stats(scene_name, completed_jobs_stats_per_scene[scene_name])

        time.sleep(5)
    
    total_duration = time.perf_counter() - total_start_perf
    
    print("-" * 40)
    print("All jobs have been processed.")
    print(f"SUMMARY: Total time taken for the entire batch: {total_duration:.2f} seconds.")
    print("-" * 40)
    
    for scene_name, stats in completed_jobs_stats_per_scene.items():
        report_stats(scene_name, stats, final=True)

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=8) as executor:
        print("Starting...")
        dispatch_jobs(jobs, executor)