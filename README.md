# Pax Case: Object and Car Make Classification

This project provides a vision pipeline for object and car make classification. It includes a command-line interface (CLI) for processing images and a Gradio-based web interface for interactive demonstrations.

## Problem Statement

The goal is to build a tool that can:
1.  **Object Classification**: Identify whether an image contains a car, truck, bicycle, or person.
2.  **Car Make Classification**: If the image contains a car, identify its make (e.g., Volkswagen, Chevrolet).

The solution must be scalable to handle up to 10 million images daily and be flexible enough to accommodate future classification tasks.

## Features

- **Two-Stage Pipeline**:
    - **Object Detection**: Uses a YOLOv8 model to detect objects (person, bicycle, car, truck).
    - **Car Make Classification**: Uses a Vision Transformer (ViT) model fine-tuned on the Stanford Cars dataset to classify the make of detected cars.
- **CLI Tool** (`cli.py`):
    - Process single images from the command line.
    - Adjustable confidence threshold for object detection.
    - Output results in JSON format to the terminal or a file.
- **Scalability and Extensibility**:
    - The architecture is designed to be scalable and extensible. See `SCALING_STRATEGY.md` for a detailed plan on scaling to 10M images/day and adding new classification types.

## Package usage
1. **Install PyPi package**
  ```bash
  pip install pax-detector
  ```
## Local Setup and Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd pax-case
    ```

2.  **Install dependencies**:
    It is recommended to use a virtual environment. This project uses `uv` for package management.
    ```bash
    uv venv 
    source .venv/bin/activate

    uv pip install -e . # To build package locally

    uv sync # If want to install local dependencies 
    ```

## Usage

### Command-Line Interface (CLI)

The CLI tool (`detector`) is used for processing individual images.

**Basic Usage**:
```bash
detector --image_path /path/to/your/image.jpg

# If using uv
uv detector --image_path /path/to/your/image.jpg
```

**Arguments**:
- `--image_path`: (Required) Path to the input image.
- `--output`: (Optional) Path to save the JSON output file.
- `--confidence`: (Optional) Confidence threshold for object detection (default: 0.5).

**Example**:
```bash
detector --image_path datasets/car-camera/images/00001.jpg --confidence 0.4 --output results.json

# If using uv
uv detector --image_path datasets/car-camera/images/00001.jpg --confidence 0.4 --output results.json
```

This will process the image, save the output to `results.json`, and print the path to the output file.

### Gradio Web Demo

The Gradio demo (`gradio_demo.py`) provides an interactive web interface to test the pipeline.

**To run the demo**:
```bash
python gradio_demo.py
```

This will start a local web server. Open your browser and navigate to the URL provided (usually `http://127.0.0.1:7860`) to access the interface.

## Extensibility

For details on how to add more classification types (e.g., helmet color, t-shirt color), please refer to the `SCALING_STRATEGY.md` document, which outlines the proposed architecture for extending the pipeline.
