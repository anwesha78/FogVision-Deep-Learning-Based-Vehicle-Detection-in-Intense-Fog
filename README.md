# FogVision-Deep-Learning-Based-Vehicle-Detection-in-Intense-Fog
FogVision is a real-time foggy object detection system built using YOLOv8 and computer vision techniques for low-visibility traffic environments. The project focuses on robust object detection under fog, haze, and degraded visibility conditions using deep learning–based feature extraction and optimized detection pipelines.

The system supports both image and video inference through a Flask-based web application and performs real-time detection of traffic objects such as vehicles, riders, pedestrians, and road obstacles.

Overview:

Low-visibility environments significantly reduce the reliability of traditional vision systems used in traffic monitoring and intelligent transportation. FogVision addresses this challenge by training YOLOv8 models directly on real-world foggy datasets instead of relying entirely on artificially enhanced imagery.


The project explores multiple experimental stages including:

Baseline YOLOv8n training
Augmentation-based optimization
CLAHE-based enhancement experiments
Optimized YOLOv8s training
Real-time frontend deployment
Lightweight post-detection enhancement
The final pipeline prioritizes robust real-world detection consistency while maintaining efficient inference speed.

Technologies Used:

Deep Learning
YOLOv8
CNN (Convolutional Neural Networks)
DNN (Deep Neural Networks)
Transfer Learning
Optimization & Training
Gradient Descent
Backpropagation
Cosine Learning Rate Scheduling
Data Augmentation
Computer Vision
OpenCV
CLAHE Enhancement
Non-Maximum Suppression (NMS)
Deployment
Flask
HTML/CSS
Python
Models Used
Model	Purpose
YOLOv8n	Baseline lightweight experimentation
YOLOv8s	Final optimized detection model

YOLOv8s was selected for the final pipeline due to:
stronger feature extraction
improved multi-scale learning
better localization performance
higher detection robustness under foggy conditions
Dataset

The model was trained on real-world foggy traffic datasets containing multiple object classes including:
Car
Bus
Truck
Rider
Bicycle
Motorcycle
Person
Train

The dataset included:
dense fog conditions
varying visibility levels
real traffic scenes
multi-object environments

Dataset splitting was performed for:

training
validation
testing
Training Strategy

Training was performed using Google Colab with NVIDIA GPU acceleration.

Baseline Training
YOLOv8n
50 epochs
640 × 640 image size
Optimized Training
YOLOv8s
50 epochs
Batch size: 16
Cosine LR scheduling enabled
Mosaic augmentation
HSV augmentation
Scaling and flipping
close_mosaic = 10

The optimized model demonstrated improved:
mAP
precision
localization quality
robustness under low visibility
Enhancement Experiments


CLAHE-based enhancement experiments were conducted to improve visibility and local contrast under foggy conditions.

However, aggressive enhancement introduced feature inconsistencies and domain shift between training and inference distributions. As a result, the final optimized pipeline focused on training directly on original foggy images while applying lightweight enhancement only for visualization purposes.

The project also proposes future integration of diffusion-based dehazing networks for improved restoration under dense fog environments.

System Workflow
User uploads image or video
OpenCV processes frames/media
YOLOv8 performs object detection
Confidence threshold filtering applied
NMS removes duplicate detections
Lightweight enhancement applied
Annotated output generated
Results displayed on frontend
Evaluation Metrics

The system was evaluated using:

Precision
Recall
F1 Score
IoU
mAP@50
mAP@50:95

These metrics were used to measure:

object localization accuracy
detection robustness
overall detection performance
Frontend

The frontend interface was developed using Flask, HTML, and CSS.

Features include:

image/video upload
real-time inference
before/after visualization
responsive UI
processed output rendering
Running the Project

Install dependencies:

pip install flask ultralytics opencv-python

Run the application:
python app.py

Open in browser:
http://127.0.0.1:5500

## Interface Preview
<img width="602" height="416" alt="image" src="https://github.com/user-attachments/assets/c2a8eba4-2607-43bf-b449-c63f28737734" />

## Detection Results
<img width="785" height="734" alt="image" src="https://github.com/user-attachments/assets/a7a9217d-66a6-4ac8-9420-1925be1968b1" />

## Training Metrics
<img width="675" height="437" alt="image" src="https://github.com/user-attachments/assets/f07229c7-93fe-4499-9809-3975cc26eb5d" />


Future Scope:

diffusion-based dehazing networks
transformer-based detection architectures
larger fog-specific datasets
edge-device deployment
autonomous driving integration
joint enhancement and detection frameworks

Author
Anwesha Sahoo
