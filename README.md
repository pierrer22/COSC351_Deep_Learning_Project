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
- Bounding boxes converted from COCO format → `(x1, y1, x2, y2)`
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

---

## Training

- Optimizer: Adam  
- Loss:
  - bounding box regression loss  
  - classification loss  

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

---

## TensorBoard

Training was monitored using TensorBoard, including:
- loss curves  
- training progress  

Screenshots are available in the `/screenshots` directory.