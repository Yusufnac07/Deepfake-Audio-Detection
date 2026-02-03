import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

DATASET_PATH = "raw_audio_dataset" 
OUTPUT_PATH = "processed_spectrograms"
MAX_FILES = 1000 

def create_spectrogram(audio_file, image_file):
    try:
        y, sr = librosa.load(audio_file, duration=3) 
        
        plt.figure(figsize=(4, 4))
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        librosa.display.specshow(S_dB, sr=sr)
        plt.axis('off') 
        plt.savefig(image_file, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"[ERROR] Failed to process: {audio_file} - {e}")

def process_data():
    categories = ["REAL", "FAKE"] 
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for category in categories:
        path = os.path.join(DATASET_PATH, category)
        save_path = os.path.join(OUTPUT_PATH, category)
        
        if not os.path.exists(path):
            print(f"[WARNING] Folder not found: '{path}'")
            print(f"Please create a folder named '{DATASET_PATH}' and put '{category}' folder inside it.")
            continue

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        print(f"[INFO] Processing category: {category}...")
        
        files = os.listdir(path)
        count = 0
        
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                if count >= MAX_FILES:
                    break
                
                audio_path = os.path.join(path, file)
                image_name = os.path.splitext(file)[0] + ".png"
                image_path = os.path.join(save_path, image_name)
                
                if not os.path.exists(image_path): 
                    create_spectrogram(audio_path, image_path)
                
                count += 1
                
                if count % 100 == 0:
                    print(f" -> {count} images processed for {category}")

if __name__ == "__main__":
    process_data()
    print("[SUCCESS] Data preprocessing completed!")