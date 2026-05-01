import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

# Paths
BASE_DIR = r"c:\Users\gupta\smart_road_safety_system"
CNN_MODEL_PATH = os.path.join(BASE_DIR, "outputs", "custom_helmet_cnn.keras")
VAL_DIR = os.path.join(BASE_DIR, "data", "classification", "val")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "outputs", "helmet_detection_scratch", "weights", "best.pt")
YOLO_DATA_YAML = os.path.join(BASE_DIR, "data", "data.yaml")

# CNN Settings
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

def evaluate_cnn():
    print("="*50)
    print("Evaluating Custom CNN Model...")
    print("="*50)
    
    if not os.path.exists(CNN_MODEL_PATH):
        print(f"Error: CNN model not found at {CNN_MODEL_PATH}")
        return

    # To avoid verbose TF logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    try:
        model = tf.keras.models.load_model(CNN_MODEL_PATH)
        print("CNN Model loaded successfully.")
    except Exception as e:
        print(f"Error loading CNN model: {e}")
        return
    
    try:
        val_ds = tf.keras.utils.image_dataset_from_directory(
            VAL_DIR,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            shuffle=False # Keep order for metrics
        )
    except Exception as e:
        print(f"Error loading validation dataset: {e}")
        return
        
    class_names = val_ds.class_names
    print(f"Classes: {class_names}")
    
    y_true = []
    y_pred_probs = []
    
    print("Generating predictions on validation set...")
    for images, labels in val_ds:
        y_true.extend(labels.numpy())
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds)
        
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('CNN Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(OUTPUT_DIR, "cnn_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}\n")

def evaluate_yolo():
    print("="*50)
    print("Evaluating YOLOv8 Model...")
    print("="*50)
    
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Error: YOLO model not found at {YOLO_MODEL_PATH}")
        return

    try:
        model = YOLO(YOLO_MODEL_PATH)
        print("YOLO Model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    
    # Run validation
    print("Running YOLO validation (this may take a few minutes depending on dataset size)...")
    try:
        metrics = model.val(
            data=YOLO_DATA_YAML,
            split='val',
            project=OUTPUT_DIR,
            name='yolo_evaluation',
            exist_ok=True
        )
        
        print(f"\nMean Average Precision (mAP50-95): {metrics.box.map:.4f}")
        print(f"Mean Average Precision (mAP50): {metrics.box.map50:.4f}")
        print(f"Mean Average Precision (mAP75): {metrics.box.map75:.4f}")
        
        print(f"\nYOLO evaluation complete. Check the {os.path.join(OUTPUT_DIR, 'yolo_evaluation')} folder for detailed PR curves, confusion matrices, and other evaluation matrices.")
    except Exception as e:
        print(f"Error during YOLO validation: {e}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    evaluate_cnn()
    evaluate_yolo()

if __name__ == "__main__":
    main()
