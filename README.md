#  Tomato Auto‑Annotation with KerasCV

## Overview

This project implements an end‑to‑end computer vision pipeline for automatic tomato detection and annotation using **TensorFlow 2.18+, Keras 3, KerasCV, and Keras Hub**.

The objective is to train an object detection model capable of identifying tomatoes in images and generating bounding box annotations. These predictions are then used to accelerate manual labelling through an **auto‑annotation workflow**.

---

## Project Goals

- Build a complete object detection pipeline
- Train a model to detect tomatoes in real-world images
- Implement an auto‑annotation system to assist labelling
- Evaluate model performance on unseen data
- Apply explainable AI and negative sampling techniques

---

## Dataset

- Custom tomato dataset annotated using **CVAT**
- Exported in **COCO format**
- Single class used:
  - `tomato`

### Design Choice

A single-class approach was used to:
- simplify the detection problem  
- improve model performance  
- maximise recall for auto‑annotation  

---

## Data Processing

- Images loaded using TensorFlow pipelines
- Bounding boxes are used in COCO format → (x, y, width, height)
- Images resized to fixed input size (512–640 resolution)
- Dataset includes:
  - shuffling
  - batching
  - prefetching

---

## Model

- **RetinaNet with ResNet50 backbone**
- Pretrained on **ImageNet**
- Fine‑tuned on tomato dataset

### Why RetinaNet?

- Good balance of accuracy and performance  
- Strong baseline for object detection  
- Works well with limited data via transfer learning  
- RetinaNet is particularly effective for handling class imbalance through focal loss, making it suitable for object detection tasks with varying object frequencies.

---

## Training

- Optimizer: SGD  
- Loss:
  - bounding box: smooth L1
  - classification: Focal Loss

- Training techniques:
  - batching and caching  
  - optional data augmentation  
  - TensorBoard monitoring  

Training shows decreasing loss across epochs, indicating successful learning.

---

## Evaluation

Model performance was evaluated using:

- loss values  
- prediction confidence  
- visual inspection of bounding boxes  

### COCO Metrics (mAP)

The model was also evaluated using COCO metrics, specifically mean Average Precision (mAP), 
which measures both detection accuracy and bounding box quality at different IoU thresholds.

The results show that the model achieves reasonable detection performance and generalises well to unseen test data.

### Results

- Accurate tomato detection in most cases  
- Some errors in:
  - dense clusters  
  - occluded objects  
  - small distant tomatoes  

---

## Explainable AI

Activation visualisations were used to highlight regions that influence model predictions. Results show the model focuses on tomato areas, confirming meaningful feature learning.

---

## Negative Sampling

Evaluation included images **without tomatoes** to test robustness.

### Outcome:
- Low false positives  
- Correct suppression of detections  

This demonstrates good real‑world reliability.

---

## Auto‑Annotation Workflow

1. Train model on labelled dataset  
2. Run inference on new images  
3. Filter predictions by confidence threshold  
4. Import predictions into **CVAT**  
5. Manually correct annotations  
6. Retrain with improved dataset  

This iterative loop significantly reduces manual annotation effort.

This creates a human-in-the-loop system where automatically generated annotations can be refined and reused to continuously improve the dataset and model performance.

---

## TensorBoard

Training was monitored using TensorBoard, including:
- loss curves  
- training progress  

Screenshots are available in the `/screenshots` directory.