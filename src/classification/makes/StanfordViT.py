import transformers
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import torch.nn as nn
from PIL import Image
import requests
from classification.base import BaseClassifier

class MakeClassifier(BaseClassifier):
    def __init__(self, model_name="therealcyberlord/stanford-car-vit-patch16", cache_dir="models"):
        self.extractor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForImageClassification.from_pretrained(model_name, cache_dir=cache_dir)

    @property
    def name(self) -> str:
        return "car_make_classification"

    def process(self, image: Image.Image, context: dict) -> dict:
        """Get top k predictions for an image."""
        k = context.get('k', 5)

        if image is None:
            raise ValueError("No image provided for classification")

        inputs = self.extractor(images=image, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = nn.functional.softmax(outputs.logits, dim=-1)

        top_k_values, top_k_indices = torch.topk(predictions, k)

        results = []
        for i in range(k):
            class_idx = top_k_indices[0][i].item()
            confidence = top_k_values[0][i].item()

            if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
                label = self.model.config.id2label[class_idx]
            else:
                label = f"Class: {class_idx}"

            results.append({
                'label': label,
                'confidence': confidence,
                'class_index': class_idx
            })

        return {'make_classification': results}


# if __name__ == "__main__":
#     classifier = MakeClassifier(model_name="therealcyberlord/stanford-car-vit-patch16", cache_dir="models")
#
#     try:
#         image = Image.open("/Users/fnayres/pax-case/datasets/car-camera/images/images/1479502700758590752.jpg")
#         context = {'k': 5}
#         result = classifier.process(image, context)
#         print("\nTop 5 predictions:")
#         for i, pred in enumerate(result['make_classification'], 1):
#             print(f"{i}. {pred['label']} - {pred['confidence']:.4f}")
#
#     except Exception as e:
#         print(f"Error: {e}")
