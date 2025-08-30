# Scaling and Extensibility Strategy

This document outlines the strategy for scaling the vision pipeline to handle 10 million images per day and for extending its capabilities with additional classification models.

## 1. Scaling to 10 Million Images per Day

Processing 10 million images daily requires a robust, distributed, and resilient architecture. A monolithic application would not be cost-effective or performant. The proposed solution is a microservices-based architecture centered around a message queue.

**Key Components**:

1.  **API Gateway**: A single entry point for all incoming image processing requests. It handles authentication, rate limiting, and routing requests to the appropriate service.

2.  **Message Queue** (e.g., RabbitMQ, AWS SQS, Google Cloud Pub/Sub):
    -   Decouples the request intake from the processing pipeline.
    -   The API Gateway publishes a message to the queue for each image, containing metadata like the image location (e.g., an S3 bucket URL).
    -   Provides persistence and allows the processing workers to consume images at their own pace, preventing overload.

3.  **Processing Workers (Microservices)**:
    -   Independent, containerized services (using Docker) that subscribe to the message queue.
    -   Each worker pulls a message, downloads the image, runs the vision pipeline, and stores the results.
    -   **Types of Workers**:
        -   `object-detection-worker`: Runs the YOLO model.
        -   `car-make-worker`: Runs the ViT model for car make classification.

4.  **Orchestration** (e.g., Kubernetes):
    -   Manages the deployment and scaling of the containerized workers.
    -   **Horizontal Pod Autoscaling (HPA)** can automatically scale the number of worker pods based on queue length or CPU/GPU utilization.
    -   This ensures that we have enough compute resources during peak loads and can scale down to save costs during off-peak hours.

5.  **Optimized Model Serving** (e.g., NVIDIA Triton Inference Server, TorchServe):
    -   Instead of loading models inside each worker, use a dedicated inference server.
    -   This allows for batching requests, concurrent model execution, and efficient GPU utilization, significantly increasing throughput.

6.  **Distributed Storage** (e.g., AWS S3, Google Cloud Storage):
    -   Store raw images and JSON results in a scalable and durable object store.

**Workflow**:

```
[Client] -> [API Gateway] -> [Message Queue] -> [Auto-Scaling Workers] -> [Results DB/Storage]
                                   ^                   |
                                   |                   v
                               [Kubernetes]      [Inference Server]
```

This architecture is cost-effective because resources are scaled based on demand. It is also resilient, as the message queue can handle request spikes and a failing worker will not cause the entire system to crash.

## 2. Adding More Classifications

The pipeline should be flexible enough to easily add new classification models without modifying existing code. This can be achieved through a modular, configuration-driven approach.

**Proposed Design**:

1.  **Classifier Abstraction**: Define a base `Classifier` class that all new classifiers must implement. This class would enforce a common interface, such as `process(image)` and `get_name()`.

    ```python
    # In a new file, e.g., classification/base.py
    from abc import ABC, abstractmethod

    class BaseClassifier(ABC):
        @abstractmethod
        def process(self, image, context: dict) -> dict:
            """Process an image and return classification results."""
            pass

        @property
        @abstractmethod
        def name(self) -> str:
            """Return the unique name of the classifier."""
            pass
    ```

2.  **Dynamic Pipeline Configuration**: The main pipeline will be driven by a configuration file (e.g., `config.yaml`) that defines the sequence of classifiers to run and any dependencies between them.

    **Example `config.yaml`**:

    ```yaml
    pipeline:
      - name: object_detection
        class: classification.objects.ObjectClassfier
        # No dependencies, runs first

      - name: car_make_classification
        class: classification.makes.StanfordViT
        depends_on: object_detection # Depends on the output of object detection
        condition: "'car' in context['object_detection']['detected_classes']"

      - name: tshirt_color_classification
        class: classification.attributes.TShirtColorClassifier
        depends_on: object_detection
        condition: "'person' in context['object_detection']['detected_classes']"
    ```

3.  **Pipeline Execution Engine**: The `VisionPipeline` will read this configuration, instantiate the required classifiers, and execute them in the correct order, passing context (e.g., results from previous steps) as needed.

**Example: Adding a T-Shirt Color Classifier**

1.  **Create the Classifier**:
    -   Develop the model and create `classification/attributes/tshirt_color.py`.
    -   Implement the `TShirtColorClassifier` class, inheriting from `BaseClassifier`.
    -   The `process` method would take the image and the bounding box of the detected 'person' from the context.

2.  **Update Configuration**:
    -   Add the new classifier to `config.yaml` as shown in the example above.

3.  **Deploy**:
    -   The pipeline will automatically load and run the new classifier for images containing a person, with no changes needed to the core `VisionPipeline` code.

This approach makes the system highly extensible and maintainable, allowing for the rapid integration of new models as business requirements evolve.
