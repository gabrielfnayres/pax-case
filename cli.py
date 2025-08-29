import os
import sys
import argparse
import json
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'classification', 'objects'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'classification', 'makes'))

from pipeline.vision_pipeline import VisionPipeline

def main():
    """
    Main function to run the CLI for the vision pipeline.
    """
    parser = argparse.ArgumentParser(description="Process an image through the vision pipeline.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
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
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_path}")
    else:
        print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
