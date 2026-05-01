import os
import cv2
import argparse
from ultralytics import YOLO

# Pre-trained YOLOv8 model
MODEL_PATH = "yolov8n.pt"
OUTPUT_DIR = r"c:\Users\gupta\smart_road_safety_system\outputs"

def check_intersection(boxA, boxB):
    # box is [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    return interArea > 0

def process_frame(frame, model):
    # Run YOLO inference
    results = model(frame, verbose=False)
    
    # Class 0: person, Class 67: cell phone (COCO dataset)
    persons = []
    phones = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if cls == 0 and conf > 0.4:
                persons.append([x1, y1, x2, y2, conf])
            elif cls == 67 and conf > 0.3:
                phones.append([x1, y1, x2, y2, conf])
                
    # Detect overlaps
    distracted_persons = []
    for person in persons:
        is_distracted = False
        for phone in phones:
            if check_intersection(person[:4], phone[:4]):
                is_distracted = True
                break
        
        if is_distracted:
            distracted_persons.append(person)
            
    # Draw results
    for person in persons:
        x1, y1, x2, y2, conf = person
        is_distracted = person in distracted_persons
        
        color = (0, 0, 255) if is_distracted else (0, 255, 0) # Red if distracted, Green otherwise
        label = "Mobile Usage Detected!" if is_distracted else "Driver"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    for phone in phones:
        x1, y1, x2, y2, conf = phone
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1) # Blue box for phone
        cv2.putText(frame, "Phone", (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    return frame

def process_image(img_path, model):
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Error reading {img_path}")
        return
        
    out_frame = process_frame(frame, model)
    out_path = os.path.join(OUTPUT_DIR, f"mobile_usage_{os.path.basename(img_path)}")
    cv2.imwrite(out_path, out_frame)
    print(f"Saved: {out_path}")

def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error reading {video_path}")
        return
        
    out_path = os.path.join(OUTPUT_DIR, f"mobile_usage_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        out_frame = process_frame(frame, model)
        out.write(out_frame)
        
    cap.release()
    out.release()
    print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Distracted Driving (Mobile Usage) Detector")
    parser.add_argument("--source", type=str, required=True, help="Path to image or video")
    args = parser.parse_args()
    
    print("Loading pre-trained YOLOv8 model for Cell Phone detection...")
    model = YOLO(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    ext = os.path.splitext(args.source)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        process_image(args.source, model)
    elif ext in ['.mp4', '.avi', '.mov']:
        process_video(args.source, model)
    else:
        print("Unsupported file format.")

if __name__ == "__main__":
    main()
