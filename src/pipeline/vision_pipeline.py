import sys 
import os
from typing import Optional
import cv2 as cv
from PIL import Image
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'src','classification', 'objects'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src','classification', 'makes'))

from classification.objects.object_classifier import ObjectClassfier
from classification.makes.StanfordViT import Classifier as MakeClassifier

class VisionPipeline:
    def __init__(self):
        """
        Initializes the vision pipeline by loading the object detection and car make classification models.
        """
        self.object_classifier = ObjectClassfier()
        self.make_classifier = MakeClassifier()

    def process_image(self, image_path):
        """
        Processes a single image through the full pipeline.

        Args:
            image_path (str): The path to the image file.

        Returns:
            list: A list of dictionaries, where each dictionary represents a detected vehicle
                  and its make classification.
        """
        img = cv.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        detections = self.object_classifier.run_detection(img, confidence=0.5)

        results = []

        for det in detections:
            if det['class_name'] in ['car', 'truck']:
                # Crop the detected vehicle from the main image
                x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
                vehicle_img_bgr = img[y1:y2, x1:x2]
                # Convert from BGR (OpenCV) to RGB (PIL)
                vehicle_img_rgb = cv.cvtColor(vehicle_img_bgr, cv.COLOR_BGR2RGB)
                pil_image = Image.fromarray(vehicle_img_rgb)

                make_predictions = self.make_classifier.get_top_k_predictions(pil_image, k=5)

                results.append({
                    'object_detection': det,
                    'make_classification': make_predictions
                })

        return results

# if __name__ == '__main__':
#     pipeline = VisionPipeline()
    
#     example_image = "/Users/fnayres/pax-case/datasets/car-camera/images/images/1479502700758590752.jpg"

#     try:
#         output = pipeline.process_image(example_image)
        
#         print(f"Processing results for: {example_image}")
#         for i, result in enumerate(output):
#             print(f"\n--- Vehicle {i+1} ---")
#             obj_det = result['object_detection']
#             print(f"  Detected: {obj_det['class_name']} with confidence {obj_det['confidence']:.2f}")
#             print(f"  Bounding Box: {obj_det['bbox']}")
            
#             print("  Top 5 Make Predictions:")
#             for pred in result['make_classification']:
#                 print(f"    - {pred['label']}: {pred['confidence']:.3f}")

#     except Exception as e:
#         print(f"An error occurred: {e}")
