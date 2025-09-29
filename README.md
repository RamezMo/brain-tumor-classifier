---
title: Brain Tumor Classification
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
---

# Brain Tumor Classification Project (97% Accuracy)
This project uses a deep learning model to classify brain tumors from MRI scans with **97% accuracy**.

## Overview
This application is the final product of a comprehensive machine learning project that involved:
- Experimenting with multiple CNN architectures (VGG16, DenseNet, ResNet50).
- Using transfer learning and advanced fine-tuning techniques to maximize performance.
- Tracking all experiments with Weights & Biases.

The final model is a fully fine-tuned **ResNet50** that can classify an MRI scan into one of four categories: `glioma`, `meningioma`, `notumor`, or `pituitary`.