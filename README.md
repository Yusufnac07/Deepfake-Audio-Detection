# 🎙️ Deepfake Audio Detection using Spectrogram Analysis & CNN

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge&logo=tensorflow)
![Librosa](https://img.shields.io/badge/Librosa-Audio%20Analysis-green?style=for-the-badge)

## 📖 Project Overview
With the rise of Generative AI, "Deepfake" voice cloning has become a major cybersecurity threat. This project aims to distinguish between **Real Human Voices** and **AI-Generated (Deepfake) Voices** by converting audio signals into visual **Mel-Spectrograms** and analyzing them using **Convolutional Neural Networks (CNN)**.

Traditional methods analyze raw audio waveforms, which is computationally expensive. Our approach converts audio into images (spectrograms) to leverage the pattern recognition power of computer vision.

## 📊 Dataset
Due to file size limitations, the dataset is not included directly in this repository. 
* **Source:** [Kaggle - DeepVoice: Deepfake Voice Recognition](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition)
* **Setup:** Download the dataset and place the `REAL` and `FAKE` folders into a directory named `raw_audio_dataset/` in the project root.

## 🎯 Purpose & Methodology
* **Visualizing Sound:** Audio files are converted into Mel-Spectrograms to reveal hidden frequency artifacts left by AI generation models.
* **Deep Learning:** A custom CNN model classifies these spectrogram images as "REAL" or "FAKE".
* **High Accuracy:** The model achieved **91.67% accuracy** on the validation set.

## 🛠️ Tech Stack & Architecture

### Data Processing Pipeline
1. **Sampling:** First 3 seconds of audio are extracted.
2. **Transformation:** Converted to Mel-Spectrograms using `Librosa`.
3. **Resizing:** Images are resized to 128x128 pixels (RGB).

### CNN Model Architecture
* **Input Layer:** 128x128x3 Image.
* **Feature Extraction:** 3 Convolutional Blocks (32, 64, 128 filters) with Max Pooling.
* **Classification:** Flatten layer followed by Dense layers with 50% Dropout to prevent overfitting.
* **Output:** Sigmoid activation for binary classification (Real vs. Fake).

## 📂 Project Structure

```bash
├── raw_audio_dataset/          # (User created) Place REAL and FAKE folders here
├── preprocessing/
│   └── audio_to_spectrogram.py # Converts .wav/.mp3 files to Spectrogram images
├── analysis/
│   └── difference_analysis.py  # Visualizes the difference between Real vs Fake spectrograms
├── model/
│   └── train_cnn_model.py      # Trains the CNN model using TensorFlow
├── inference/
│   └── predict_panel.py        # Loads the trained model and predicts on new images
└── requirements.txt            # Project dependencies
```

🚀 Installation & Usage
1. Prerequisites

Install the required Python libraries:
    
    pip install -r requirements.txt

2. Workflow

Step 1: Data Preparation Place your audio files in a dataset folder and run the preprocessing script to generate spectrogram images.

    python preprocessing/audio_to_spectrogram.py

Step 2: Analysis (Optional) Visualize the visual differences between real and fake audio fingerprints.

    python analysis/difference_analysis.py

Step 3: Training the Model Train the CNN model on the generated spectrograms.

    python model/train_cnn_model.py

This will save the trained model as deepfake_ses_modeli.keras.

Step 4: Testing & Prediction Run the prediction panel to test the model on random samples.

    python inference/predict_panel.py

📊 Results
    Training Accuracy: 90.38%
    Validation Accuracy: 91.67%
    Loss: 0.44

The model successfully identifies micro-frequency errors and "sharp cut" artifacts typical of AI-generated audio, which are often inaudible to the human ear.
👥 Contributor
    Yusuf Can GÖREN

