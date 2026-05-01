# Smart Road Safety System 🚦

An end-to-end AI-powered Road Safety dashboard designed to automatically detect two primary traffic violations: **Helmet Usage** and **Distracted Driving (Mobile Phone Usage)**.

This full-stack application uses a hybrid machine-learning approach, leveraging a custom-trained Convolutional Neural Network (CNN) alongside the state-of-the-art YOLOv8 object detection model, all wrapped in a premium React user interface.

---

## 🌟 Key Features

1. **Helmet Detection (Custom CNN)**
   - Utilizes a custom TensorFlow/Keras sequential CNN model built entirely from scratch.
   - Detects riders and mathematically crops the upper body/head region before passing it to the CNN.
   - Accurately classifies whether the rider is wearing a helmet or not.

2. **Distracted Driving Detection (YOLOv8)**
   - Leverages pre-trained YOLO weights (COCO dataset) to actively scan for `Person` and `Cell Phone` classes.
   - Computes bounding box intersections to detect if a driver is actively using a mobile phone.

3. **Live Dashboard (React + Vite)**
   - A stunning, vibrant, MindHero-inspired web dashboard.
   - Streams live, annotated MJPEG video directly from the backend.
   - Displays real-time metrics including total vehicles scanned, helmet violations, and distracted drivers.

---

## 🏗️ Architecture

This project is separated into three distinct layers:

1. **Machine Learning (`src/`)**: Contains all the research, data preparation, and training scripts for the custom CNN helmet classifier.
2. **Backend API (`backend/`)**: A standalone Python Flask server that loads the `.keras` and `.pt` ML models and serves live metrics and MJPEG video streams over HTTP.
3. **Frontend UI (`frontend/`)**: A fast, modern React web application built with Vite, styled with custom CSS glassmorphism aesthetics.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js & npm

### One-Click Startup (Windows)
The absolute easiest way to run the entire system is to simply double-click the `start.bat` file located in the root directory. 
This will automatically:
1. Start the Flask AI Backend on `http://localhost:5000`
2. Start the React Frontend on `http://localhost:5173`

### Manual Startup

**1. Start the Backend:**
```bash
cd backend
pip install -r requirements.txt
python app.py
```

**2. Start the Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your web browser to view the live dashboard!

---

## 📂 Project Structure

```text
smart_road_safety_system/
├── backend/                  # Standalone Flask API
│   ├── app.py                # Main server script handling ML inference
│   └── requirements.txt      # API-specific dependencies
├── frontend/                 # React Web Dashboard (Vite)
│   ├── src/App.jsx           # Main UI logic
│   └── src/index.css         # Custom MindHero styling
├── src/                      # ML Training Scripts
│   ├── prep_data.py          # VOC to YOLO conversion script
│   └── train_custom_cnn.py   # TensorFlow CNN architecture and training
├── test/                     # Test data for the live feed
├── outputs/                  # Saved .keras custom models
├── yolov8n.pt                # YOLO weights
└── start.bat                 # 1-Click launcher
```
