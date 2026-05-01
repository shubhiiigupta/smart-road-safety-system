from ultralytics import YOLO
import os

def main():
    # Define absolute paths
    data_yaml_path = r"c:\Users\gupta\smart_road_safety_system\data\data.yaml"
    
    # Initialize a new model from scratch using the YOLOv8n architecture (no pretrained weights)
    print("Initializing YOLOv8n model from scratch...")
    model = YOLO("yolov8n.yaml")
    
    # Train the model
    print("Starting training...")
    results = model.train(
        data=data_yaml_path,
        epochs=100,        # 100 epochs since we are training from scratch
        imgsz=640,
        batch=16,
        project=r"c:\Users\gupta\smart_road_safety_system\outputs",
        name="helmet_detection_scratch"
    )
    
    print("Training complete! Model saved in outputs/helmet_detection_scratch")

if __name__ == "__main__":
    main()
