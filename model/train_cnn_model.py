import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

DATA_PATH = "processed_spectrograms" 
MODEL_SAVE_PATH = "saved_models"
RESULTS_PATH = "results"
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 10

def train_model():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at '{DATA_PATH}'. Run preprocessing script first.")
        return

    print("[INFO] Loading dataset...")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    print("[INFO] Building CNN Model...")
    model = Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        
        Conv2D(128, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), 
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    print(f"[INFO] Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
        
    model_file = os.path.join(MODEL_SAVE_PATH, "deepfake_audio_detector.keras")
    model.save(model_file)
    print(f"[SUCCESS] Model saved to: {model_file}")
    
    plot_results(history)

def plot_results(history):
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(8, 8))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    save_file = os.path.join(RESULTS_PATH, "training_graph.png")
    plt.savefig(save_file)
    print(f"[INFO] Training graph saved to {save_file}")
    plt.show()

if __name__ == "__main__":
    train_model()