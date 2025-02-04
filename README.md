# Sentiment & Emotion Classification

## Project ID
**Course:** CIS 480
**Assignment:** Senior Project
**Instructor:** Dr. Grabowski
**Student Name:** Brendon Vineyard
**Date:** 02/04/2025

## Overview
This project involves processing and analyzing a dataset using machine learning techniques. It includes data preprocessing, model training, and evaluation using both image and text datasets.

## Datasets
- **FER 2013** (Facial Expression Recognition 2013)
    - 7 emotion categories: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
    - Train/Test split: Pre-organized folders
- **Sentiment140** (Twitter Sentiment Analysis)
    - 160,000 training samples, 40,000 test samples
    - Labels: 0 (Negative), 2 (Neutral), 4 (Positive)

## Project Structure
content/

## Installation
N/A

## Steps to Run the Project
N/A

## Requirements
- Python 3.8+
- TensorFlow 2.18+
- PyTorch 2.5+
- NumPy, Pandas, OpenCV, Matplotlib

## GPU Support
Ensure Tensorflow and PyTorch recognize your GPU:
    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```

## Troubleshooting
- **GPU Not Detected**
    - Check CUDA installation (`nvcc --version`)
    - Check PyTorch & TensorFlow compatibility (`torch.version.cuda` & `tf.test.is_gpu_available()`)
- **Missing Libraries**
    - Run `pip install -r n/a`

## License
N/A?