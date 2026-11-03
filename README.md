# yolo-object-detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![Label Studio](https://img.shields.io/badge/Label_Studio-Annotated-green.svg)](https://labelstud.io/)

This project develops an AI object detection model utilizing the YOLO architecture. The custom dataset was built and annotated entirely from scratch using Label Studio to analyze and evaluate model performance.

## Key Features
* **Custom Dataset Annotation:** Created and managed custom datasets using Label Studio.
* **Model Training & Optimization:** Trained the YOLO model and tuned hyperparameters to maximize detection accuracy.
* **Performance Evaluation:** Evaluated the model using standard metrics including mAP, Precision, and Recall.

## Tech Stack
* **Deep Learning Framework:** PyTorch, Ultralytics (YOLO)
* **Data Processing & Annotation:** Label Studio, OpenCV
* **Environment:** Python

## Repository Structure
```text
├── configs/               # .yaml configuration files for the dataset
├── src/                   # Source code directory
│   ├── train.py           # Script for model training
│   └── inference.py       # Script for running predictions
├── requirements.txt       # List of required dependencies
└── README.md
