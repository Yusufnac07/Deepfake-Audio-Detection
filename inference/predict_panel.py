import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

MODEL_PATH = os.path.join("saved_models", "deepfake_audio_detector.keras")
TEST_DATA_PATH = "processed_spectrograms"
IMG_SIZE = (128, 128)

def predict_and_visualize():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}. Train the model first.")
        return

    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    all_images = []
    for root, dirs, files in os.walk(TEST_DATA_PATH):
        for file in files:
            if file.endswith(".png"):
                all_images.append(os.path.join(root, file))
    
    if len(all_images) < 4:
        print("[ERROR] Not enough images to test. Need at least 4.")
        return

    sample_images = random.sample(all_images, 4)
    
    plt.figure(figsize=(15, 10))
    plt.suptitle("AI Deepfake Audio Detection Panel", fontsize=20)

    for i, img_path in enumerate(sample_images):
        img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.0

        prediction = model.predict(img_array, verbose=0)
        score = prediction[0][0]
        
        is_real_file = "REAL" in img_path.upper()
        gercek_etiket = "REAL" if is_real_file else "FAKE"
        
        is_prediction_real = score > 0.5
        tahmin_etiket = "GERÇEK (REAL)" if is_prediction_real else "SAHTE (FAKE)"
        guven = score if is_prediction_real else 1 - score
        
        if (is_real_file and is_prediction_real) or (not is_real_file and not is_prediction_real):
            color = "green"
        else:
            color = "red"
        
        plt.subplot(1, 4, i+1)
        plt.imshow(img)
        plt.title(f"Prediction: {tahmin_etiket}\n(Conf: %{guven*100:.1f})\nTruth: {gercek_etiket}", 
                  color=color, fontsize=12, fontweight='bold')
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict_and_visualize()