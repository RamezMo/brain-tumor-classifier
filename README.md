---
title: Brain Tumor Classification
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Brain Tumor Classification Project

This project uses a deep learning model to classify brain tumors from MRI scans with **97% accuracy**.

## Overview
This application is the final product of a comprehensive machine learning project that involved:
- Experimenting with multiple CNN architectures (VGG16, DenseNet, ResNet50).
- Using transfer learning and advanced fine-tuning techniques to maximize performance.
- Tracking all experiments with Weights & Biases.

The final model is a fully fine-tuned `ResNet50` that can classify an MRI scan into one of four categories:
- Glioma
- Meningioma
- No Tumor
- Pituitary

## How to Use
1. Click "Choose a file" to upload a brain MRI scan.
2. Click the "Predict" button.
3. The model will analyze the image and display the predicted tumor type and a confidence score.