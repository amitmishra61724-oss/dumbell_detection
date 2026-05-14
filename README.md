🏋️ Dumbbell Detection using YOLOv9
This repository provides a complete pipeline for detecting dumbbells using the state-of-the-art YOLOv9 architecture. It includes scripts for training, evaluation, and real-time inference.

🚀 Getting Started
1. Installation
First, clone the repository and install the required dependencies. You will need Python 3.8+ and PyTorch.

pip install ultralytics
2. Dataset Configuration
Your dataset should be in YOLO format. Create a data.yaml file with the following structure:

train: ./train/images
val: ./val/images
test: ./test/images

nc: 1
names: ['dumbbell']
🏗️ Model Training
We use the YOLOv9-C (compact) model as a backbone, fine-tuning it on the specific dumbbell classes for 100 epochs.

from ultralytics import YOLO

# Load a pretrained YOLOv9 model
model = YOLO('yolov9c.pt')

# Start training
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='dumbbell_optimized'
)
📊 Evaluation & Metrics
After training, the model is evaluated on the test split to calculate the Mean Average Precision (mAP).

Metric	Value
mAP@50	Calculated during validation
mAP@50-95	Calculated during validation
metrics = model.val(split='test')
print(f'Final mAP50: {metrics.box.map50}')
📸 Inference / Prediction
To run the model on your own images or videos:

# Run detection
results = model.predict(source='your_image.jpg', save=True, conf=0.25)
🛠️ Built With
[redacted link] - The core detection framework.
PyTorch - Deep learning backend.
Google Colab - Used for GPU-accelerated training.
📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

