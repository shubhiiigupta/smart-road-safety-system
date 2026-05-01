import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = r"c:\Users\gupta\smart_road_safety_system\outputs\custom_helmet_cnn.keras"
VAL_DIR = r"c:\Users\gupta\smart_road_safety_system\data\classification\val"
OUTPUT_IMG = r"c:\Users\gupta\smart_road_safety_system\outputs\predictions.png"

IMG_HEIGHT = 128
IMG_WIDTH = 128

def main():
    print("Loading the trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Get a list of all validation images
    image_paths = []
    classes = ['With_Helmet', 'Without_Helmet']
    
    for cls in classes:
        cls_dir = os.path.join(VAL_DIR, cls)
        for img_name in os.listdir(cls_dir):
            if img_name.endswith('.png'):
                image_paths.append((os.path.join(cls_dir, img_name), cls))
                
    if not image_paths:
        print("No validation images found.")
        return
        
    # Randomly select 6 images
    random.shuffle(image_paths)
    selected = image_paths[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (img_path, true_cls) in enumerate(selected):
        # Load and preprocess image
        img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        
        # Predict
        prediction = model.predict(img_array, verbose=0)
        score = prediction[0][0]
        
        # In our training set (due to image_dataset_from_directory alphabetical sorting):
        # Class 0: With_Helmet
        # Class 1: Without_Helmet
        pred_cls = 'Without_Helmet' if score >= 0.5 else 'With_Helmet'
        confidence = score if score >= 0.5 else 1 - score
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        color = 'green' if pred_cls == true_cls else 'red'
        title = f"Pred: {pred_cls}\nTrue: {true_cls}\nConf: {confidence:.2f}"
        axes[i].set_title(title, color=color)
        
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"Predictions saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()
