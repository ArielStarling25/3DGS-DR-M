import os
import re
import argparse
from pathlib import Path
import imageio, cv2

# Easily expandable list of keywords to ignore in the GT directory
IGNORE_IDENTIFIERS = ['normal', 'alpha', 'disp']
# Supported image extensions
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

def extract_number(filename):
    """
    Extracts the first sequence of digits from a filename to use for sorting.
    Returns -1 if no numbers are found.
    """
    match = re.search(r'\d+', filename.stem)
    return int(match.group()) if match else -1

def create_gif(image_paths, output_path, fps=24):
    """
    Reads a sorted list of images and compiles them into a high-quality .gif
    using imageio and OpenCV.
    """
    if not image_paths:
        print(f"Warning: No images found to create {output_path.name}")
        return
    print(f"Generating {output_path.name} with {len(image_paths)} frames at {fps} FPS...")
    with imageio.get_writer(output_path, mode='I', fps=fps, loop=0) as writer:
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Warning: Could not read {image_path.name}. Skipping.")
                continue
            # Convert BGR (OpenCV's default) to RGB (imageio/GIF format)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)
    print(f"Successfully saved to: {output_path}\n")

def generate_comparison_gifs(gt_dir, renders_dir, output_dir):
    gt_dir = Path(gt_dir)
    renders_dir = Path(renders_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_images = []
    if gt_dir.exists():
        for file_path in gt_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
                # Filter out files containing any of the ignore identifiers
                if not any(keyword in file_path.stem.lower() for keyword in IGNORE_IDENTIFIERS):
                    gt_images.append(file_path)
        
        # Sort using the numerical value found in the filename (e.g., 'r_10' -> 10)
        gt_images.sort(key=extract_number)
    else:
        print(f"Error: GT directory '{gt_dir}' does not exist.")

    render_images = []
    if renders_dir.exists():
        for file_path in renders_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
                render_images.append(file_path)
        
        # Sort using the numerical value (e.g., '00001' -> 1)
        render_images.sort(key=extract_number)
    else:
        print(f"Error: Renders directory '{renders_dir}' does not exist.")

    # Generate GIFs
    create_gif(gt_images, output_dir / 'gt_view.gif')
    create_gif(render_images, output_dir / 'render_view.gif')
    
    print(f"Process complete. Check the '{output_dir}' directory for your GIFs.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate GIFs from GT and Render image directories.")
    parser.add_argument('--gt', dest='gt_image_directory', required=True, help="Directory containing Ground Truth images")
    parser.add_argument('--renders', dest='renders_directory', required=True, help="Directory containing Render images")
    parser.add_argument('--out', dest='output_directory', required=True, help="Directory to output the generated GIFs")
    
    args = parser.parse_args()
    
    generate_comparison_gifs(
        args.gt_image_directory, 
        args.renders_directory, 
        args.output_directory
    )