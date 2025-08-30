import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from pipeline.vision_pipeline import VisionPipeline

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main():
    """
    Main function to run the CLI for the vision pipeline.
    """
    parser = argparse.ArgumentParser(description="Process an image through the vision pipeline.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output", type=str, help="Path to save the JSON output.", default=None)
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for object detection.")

    args = parser.parse_args()

    try:
        pipeline = VisionPipeline()
    except Exception as e:
        print(f"Error initializing the pipeline: {e}")
        return

    try:
        results = pipeline.process_image(args.image_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)
        print(f"Results saved to {output_path}")
    else:
        print(json.dumps(results, indent=4, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
