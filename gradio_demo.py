import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
import sys
import os
from pathlib import Path


sys.path.append(os.path.join(os.path.dirname(__file__), 'src','classification', 'objects'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src','classification', 'makes'))

from object_classifier import ObjectClassfier
from StanfordViT import Classifier as CarMakeClassifier


class IntegratedPipeline:
    """
    Integrated pipeline that combines object detection and car make classification
    """
    
    def __init__(self, cache_dir="models"):
        self.object_classifier = ObjectClassfier(model_version="yolov8l.pt", cache_dir=cache_dir) 
        self.car_make_classifier = CarMakeClassifier(model_name="therealcyberlord/stanford-car-vit-patch16", cache_dir=cache_dir)
        
    def process_image(self, image):
        """
        Process an image through the complete pipeline
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            dict: Results containing object detection and car make classification (if applicable)
        """
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
            
        results = self.object_classifier.detect(image, save_results=False, confidence=0.3)
        detections = self.object_classifier.process_detection(results)
        
        response = {
            'object_detections': detections,
            'car_make_predictions': None,
            'summary': '',
            'annotated_image': None
        }
        
        # Create annotated image
        if isinstance(image, np.ndarray):
            annotated_image = image.copy()
        else:
            annotated_image = np.array(image_pil)
            
        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Draw rectangle
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(annotated_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        response['annotated_image'] = annotated_image
        
        car_detections = [det for det in detections if det['class_name'] == 'car']
        
        if car_detections:
            try:
                # Use the full image for car make classification
                car_predictions = self.car_make_classifier.get_top_k_predictions(image_pil, k=5)
                response['car_make_predictions'] = car_predictions
                
                # Create summary
                top_prediction = car_predictions[0]
                response['summary'] = f"🚗 Car detected! Top prediction: {top_prediction['label']} (confidence: {top_prediction['confidence']:.3f})"
                
            except Exception as e:
                response['summary'] = f"🚗 Car detected, but car make classification failed: {str(e)}"
        else:
            if detections:
                detected_objects = [det['class_name'] for det in detections]
                unique_objects = list(set(detected_objects))
                response['summary'] = f"🔍 Detected objects: {', '.join(unique_objects)}. No cars found for make classification."
            else:
                response['summary'] = "❌ No objects detected in the specified categories (person, bicycle, car, truck)."
        
        return response


def create_gradio_interface():
    """
    Create and configure the Gradio interface
    """
    
    pipeline = IntegratedPipeline(cache_dir="models")
    
    def process_and_display(image):
        """
        Process image and return formatted results for Gradio
        """
        if image is None:
            return "Please upload an image", None, "No results", "No results"
        
        try:
            results = pipeline.process_image(image)
            
            object_results = "## Object Detection Results\n\n"
            if results['object_detections']:
                for i, det in enumerate(results['object_detections'], 1):
                    object_results += f"**Detection {i}:**\n"
                    object_results += f"- Class: {det['class_name']}\n"
                    object_results += f"- Confidence: {det['confidence']:.3f}\n"
                    object_results += f"- Bounding Box: [{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, {det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]\n\n"
            else:
                object_results += "No objects detected in the specified categories.\n"
            
            car_results = "## Car Make Classification Results\n\n"
            if results['car_make_predictions']:
                car_results += "**Top 5 Predictions:**\n\n"
                for i, pred in enumerate(results['car_make_predictions'], 1):
                    car_results += f"{i}. **{pred['label']}** - {pred['confidence']:.3f}\n"
            else:
                car_results += "No car make classification performed (no cars detected).\n"
            
            return (
                results['summary'],
                results['annotated_image'],
                object_results,
                car_results
            )
            
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            return error_msg, None, error_msg, error_msg
    
    with gr.Blocks(title="Object & Car Make Classification Pipeline", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚗 Object & Car Make Classification Pipeline
        
        This demo showcases a two-stage classification pipeline:
        1. **Object Detection**: Identifies objects (person, bicycle, car, truck) using YOLO
        2. **Car Make Classification**: If a car is detected, classifies the car make using a Stanford ViT model
        
        Upload an image to see the pipeline in action!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    height=400
                )
                
                process_btn = gr.Button(
                    "🔍 Analyze Image", 
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=1):
                annotated_output = gr.Image(
                    label="Detection Results",
                    height=400
                )
        
        summary_output = gr.Textbox(
            label="Summary",
            lines=2,
            max_lines=3
        )
        
        with gr.Row():
            with gr.Column():
                object_results = gr.Markdown(
                    label="Object Detection Details",
                    value="Upload an image to see object detection results"
                )
            
            with gr.Column():
                car_results = gr.Markdown(
                    label="Car Make Classification Details", 
                    value="Upload an image to see car make classification results"
                )
        
        
        process_btn.click(
            fn=process_and_display,
            inputs=[image_input],
            outputs=[summary_output, annotated_output, object_results, car_results]
        )
        
        image_input.change(
            fn=process_and_display,
            inputs=[image_input],
            outputs=[summary_output, annotated_output, object_results, car_results]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    
    print("🚀 Starting Gradio interface...")
    print("📝 Pipeline features:")
    print("   - Object detection (person, bicycle, car, truck)")
    print("   - Car make classification for detected cars")
    print("   - Real-time image annotation")
    print("   - Top-5 car make predictions")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  
        show_error=True
    )
