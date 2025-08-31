import json
import argparse
from pathlib import Path
import numpy as np

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
    parser = argparse.ArgumentParser(description="Detect objects and classify car makes in an image.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_path', type=str, help='Path to a single image file.')
    group.add_argument('--images_dir', type=str, help='Path to a directory of image files.')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for object detection.')

    args = parser.parse_args()

    pipeline = VisionPipeline()
    results = {}

    try:
        if args.image_path:
            if not Path(args.image_path).is_file():
                print(f"Error: Image file not found at {args.image_path}")
                return
            results = pipeline.process_image(args.image_path, args.confidence)
        elif args.images_dir:
            if not Path(args.images_dir).is_dir():
                print(f"Error: Directory not found at {args.images_dir}")
                return
            results = pipeline.process_multiple_images(args.images_dir, args.confidence)
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return

    print(json.dumps(results, cls=NumpyEncoder, indent=4))

if __name__ == '__main__':
    main()
