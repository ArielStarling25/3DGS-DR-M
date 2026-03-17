import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
from datetime import datetime

# 1. Updated training_list to use dictionaries like CODE 2
training_list = [
    {"name": "ball", "group": "refnerf", "iteration": 50000},
    {"name": "car", "group": "refnerf", "iteration": 50000},
    {"name": "coffee", "group": "refnerf", "iteration": 50000},
    {"name": "helmet", "group": "refnerf", "iteration": 50000},
    {"name": "teapot", "group": "refnerf", "iteration": 50000},
    {"name": "toaster", "group": "refnerf", "iteration": 50000}
]

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
    
    # Construct paths using dictionary values
    dataset_path = os.path.join(project_root, "data", scene['group'], scene['name'])
    output_directory = os.path.join(output_dir, scene['group'], scene['name'])
    set_iterations = scene['iteration']
    
    print("Dataset Path set to: ", dataset_path)
    print("Output Directory set to: ", output_directory)

    train_duration = 0.0
    mesh_duration = 0.0

    # Train
    cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 train.py -s {dataset_path} -m {output_directory} --eval --iterations {set_iterations} --white_background"
    print(cmd)
    start_train = time.perf_counter()
    if not dry_run:
        os.system(cmd)
    train_duration = time.perf_counter() - start_train

    # Extract PLY
    cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 extract_ply_mesh.py --white_background --model_path {output_directory} --iteration {set_iterations} --gaussian_ply_only"
    print(cmd)
    start_mesh = time.perf_counter()
    if not dry_run:
        os.system(cmd)
    mesh_duration = time.perf_counter() - start_mesh
    
    # Evaluation + render images
    cmd = f"OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={gpu} python3 eval.py --white_background --save_images --model_path {output_directory}"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    # Debug renders
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
        w_name, w_dur, w_sub, w_time, w_m = 25, 10, 10, 20, 8
        
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
            
            # NOTE: Update 'results.txt' if your eval.py outputs the metric file with a different name
            results_path = os.path.join(output_dir, stat['group'], scene_name, 'metric.txt')
            
            # Helper function to extract CSV metrics: psnr:val,ssim:val,lpips:val,fps:val
            def get_metrics(path):
                if os.path.exists(path):
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            # Split by comma to get key:value pairs
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

            # Fetch metrics
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
            
            print(f"Job finished. Releasing GPU {gpu}")
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