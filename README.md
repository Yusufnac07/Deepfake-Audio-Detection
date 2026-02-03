# 🎙️ Deepfake Audio Detection using Spectrogram Analysis & CNN

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge&logo=tensorflow)
![Librosa](https://img.shields.io/badge/Librosa-Audio%20Analysis-green?style=for-the-badge)

## 📖 Project Overview
With the rise of Generative AI, "Deepfake" voice cloning has become a major cybersecurity threat. This project aims to distinguish between **Real Human Voices** and **AI-Generated (Deepfake) Voices** by converting audio signals into visual **Mel-Spectrograms** and analyzing them using **Convolutional Neural Networks (CNN)**.

Traditional methods analyze raw audio waveforms, which is computationally expensive. Our approach converts audio into images (spectrograms) to leverage the pattern recognition power of computer vision.

## 🎯 Purpose & Methodology
* **Visualizing Sound:** Audio files are converted into Mel-Spectrograms to reveal hidden frequency artifacts left by AI generation models.
* **Deep Learning:** A custom CNN model classifies these spectrogram images as "REAL" or "FAKE".
* **High Accuracy:** The model achieved **91.67% accuracy** on the validation set.

## 🛠️ Tech Stack & Architecture

### Data Processing Pipeline
1.  **Sampling:** First 3 seconds of audio are extracted.
2.  **Transformation:** Converted to Mel-Spectrograms using `Librosa`.
3.  **Resizing:** Images are resized to 128x128 pixels (RGB).

### CNN Model Architecture
* **Input Layer:** 128x128x3 Image
* **Feature Extraction:** 3 Convolutional Blocks (32, 64, 128 filters) with Max Pooling.
* **Classification:** Flatten layer followed by Dense layers with 50% Dropout to prevent overfitting.
* **Output:** Sigmoid activation for binary classification (Real vs. Fake).

## 📂 Project Structure

```bash
├── preprocessing/
│   └── audio_to_spectrogram.py # Converts .wav/.mp3 files to Spectrogram images
├── analysis/
│   └── difference_analysis.py  # Visualizes the difference between Real vs Fake spectrograms
├── model/
│   └── train_cnn_model.py      # Trains the CNN model using TensorFlow
├── inference/
│   └── predict_panel.py        # Loads the trained model and predicts on new images
└── requirements.txt            # Project dependencies
