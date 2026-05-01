import os
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = r"c:\Users\gupta\smart_road_safety_system\data\classification"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
OUTPUT_DIR = r"c:\Users\gupta\smart_road_safety_system\outputs"

BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
EPOCHS = 15

def build_model():
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') # Binary classification
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

def main():
    print("Loading datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)
        
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)
        
    class_names = train_ds.class_names
    print(f"Classes: {class_names}")
    
    # Configure dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    print("Building custom CNN...")
    model = build_model()
    model.summary()
    
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_save_path = os.path.join(OUTPUT_DIR, "custom_helmet_cnn.keras")
    model.save(model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
