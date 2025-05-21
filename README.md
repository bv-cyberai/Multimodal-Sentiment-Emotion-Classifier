# Multi-Modal Sentiment and Emotion Classification

This project presents a multi-modal deep learning system that classifies both **emotion** and **sentiment** from paired **image and text** inputs. The system combines computer vision and natural language processing techniques to better understand affective content, particularly in social media contexts.

## Overview

Traditional models treat image and text inputs separately, missing important emotional context. This project fuses visual and textual signals using:

- Three CNNs trained on facial emotion datasets
- A BiLSTM model trained on tweet sentiment data
- A fusion MLP that combines outputs for joint classification

The system is showcased through an interactive Gradio interface that visualizes predictions and confidence scores for sampled dataset examples.

## Datasets

- **FER-2013** — Grayscale images with labeled facial expressions
- **RAF-DB** — Real-world affective faces
- **FER+** — Enhanced FER dataset with crowd-validated labels
- **Sentiment140** — 1.6M tweets labeled as positive or negative

## Model Architecture

- **CNNs**: Trained separately on FER-2013, RAF-DB, and FER+
- **BiLSTM**: Trained on Sentiment140 for sentiment classification
- **Fusion MLP**: Takes CNN outputs + BiLSTM output and classifies final emotion/sentiment
- **Gradio UI**: Displays image-caption pairs, predictions, ground truths, and confidence charts

## Results

- **Emotion Accuracy**: Up to **95%**
- **Sentiment Accuracy**: Ranged from **82% to 95%**
- **Fused Accuracy Estimate**: **88.5%–92.5%**

## Features

- Interactive Gradio demo
- Clean, scrollable gallery format
- Real-time confidence visualization
- Ground truth vs predicted label comparison

## Requirements

This project runs entirely within a Python notebook environment such as Google Colab or Jupyter Notebook. The following libraries must be available:

- TensorFlow
- Keras
- NumPy
- Pandas
- scikit-learn
- matplotlib
- seaborn
- Gradio

In Google Colab, you can install any missing dependencies with:

```python
!pip install gradio seaborn
```

## Running the Demo

1. Open the notebook: `MultiModalModelBrendon.ipynb`
2. Follow and run ALL the cells to:
    - Load models and datasets
    - Process and fuse image and text inputs
    - Launch the interactive Gradio interface
3. The trained models are located in the `/Needed Models/` directory if needed.
4. The demo automatically samples and visualizes predictions using a scrollable gallery of image-caption pairs.

No command-line usage is required.

## Applications

- Social media content moderation
- Sentiment trend analysis
- Mental health signal detection
- Affective computing and digital wellness

## Author

**Brendon Vineyard**  
Capstone Senior Project, SUNY Potsdam (Spring 2025)  
Advisor: Dr. Laura Grabowski  
Email: vineyabn207@potsdam.edu

---

*This project was developed to fulfill the requirements of CIS 480: Senior Project in Computer Science at SUNY Potsdam.*
