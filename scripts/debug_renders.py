import os
import re
import argparse
from pathlib import Path
from PIL import Image

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

def create_gif(image_paths, output_path, duration=100):
    """
    Loads a list of image paths and saves them as a GIF.
    Handles transparent PNGs correctly to prevent frame stacking.
    """
    if not image_paths:
        print(f"Warning: No images found to create {output_path.name}")
        return

    print(f"Loading {len(image_paths)} images for {output_path.name}...")
    
    # Open all images and ensure they are loaded in RGBA mode
    # so the alpha channel is properly recognized
    frames = [Image.open(img_path).convert("RGBA") for img_path in image_paths]
    
    # Save as GIF
    print(f"Saving {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        disposal=2
    )
    print("Done.\n")

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

    # 4. Generate GIFs
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