import os
import glob
import argparse
from simpleosmrenderer.renderer import render_osm_maps

def render_single_file(input_file, output_dir):
    """Render a single JSON file into maps"""
    print(f"Rendering {input_file} -> {output_dir}...")
    render_osm_maps(input_file=input_file, output_dir=output_dir)

def render_all_frames(input_folder="output", output_base="maps"):
    """Render all JSON files in the input folder"""
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    if not json_files:
        print("No JSON files found to render.")
        return

    for json_path in json_files:
        file_name = os.path.basename(json_path)
        scenario_name = os.path.splitext(file_name)[0]
        output_dir = os.path.join(output_base, scenario_name)
        render_single_file(json_path, output_dir)

    print("Done rendering all JSON files!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Folium maps from JSON data")
    parser.add_argument("--input_file", type=str, 
                        help="Single JSON file to render (optional)")
    parser.add_argument("--output_dir", type=str, 
                        help="Output directory for single file render (optional)")
    parser.add_argument("--input_folder", type=str, default="output",
                        help="Folder containing JSON files for batch processing")
    parser.add_argument("--output_base", type=str, default="maps",
                        help="Base directory for batch output")
    
    args = parser.parse_args()
    
    if args.input_file:
        # Single file mode
        output_dir = args.output_dir if args.output_dir else "route_output"
        render_single_file(args.input_file, output_dir)
    else:
        # Batch processing mode
        render_all_frames(args.input_folder, args.output_base)
