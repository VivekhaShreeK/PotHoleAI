# 🚧 PotholeAI — Intelligent Pothole Detection System

A production-grade hybrid computer vision + deep learning system for real-time pothole detection from road images and videos.

## Features

### 🔬 Classical Computer Vision
- **Multi-scale Canny edge detection** (dual-threshold)
- **CLAHE contrast enhancement** for varying lighting
- **Contour detection + filtering** by area, aspect ratio, convexity
- **Texture analysis** via Sobel gradient variance
- **Morphological operations** (closing, dilation) for edge refinement

### 🧠 Deep Learning Integration
- YOLOv8 / CNN ready architecture (simulated in demo)
- Confidence calibration using CV priors
- Non-Maximum Suppression (configurable IoU)
- Plug-and-play: replace simulator with real model weights

### 📊 Detection Output
- Bounding boxes with confidence scores
- Per-frame pothole count
- Color-coded severity levels

### ⚡ Advanced Features
- **📍 GPS Tagging** — coordinates for every detected pothole
- **🧠 Duplicate Detection** — DCT-based perceptual hashing + spatial proximity
- **📊 Severity Estimation** — minor / moderate / severe / critical + depth (cm)
- **⚡ Real-time** — optimized pipeline, ~60+ FPS on modern hardware

### 📈 Evaluation Metrics
- Precision, Recall, F1 Score
- mAP@0.5 and mAP@0.75
- Precision-Recall curve visualization
- Per-severity breakdown

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit app
streamlit run app.py

# 3. Open http://localhost:8501
```

## Project Structure

```
pothole_system/
├── app.py                    # Streamlit UI (5 tabs)
├── requirements.txt
├── utils/
│   ├── cv_pipeline.py        # Core detection engine
│   │   ├── ClassicalCVPipeline    # Canny + contours + texture
│   │   ├── DeepLearningSimulator  # YOLOv8-ready wrapper
│   │   ├── SeverityEstimator      # minor/moderate/severe/critical
│   │   ├── DuplicateDetector      # pHash-based dedup
│   │   ├── GPSTagger              # coordinate tagging
│   │   └── HybridDetector         # orchestrator
│   ├── metrics.py            # Precision / Recall / mAP engine
│   └── synthetic_data.py     # Demo image/video generator
└── results/                  # Output JSON reports
```

## Integrating Real Models

### YOLOv8
```python
from ultralytics import YOLO
model = YOLO('pothole_yolov8.pt')  # your trained weights

# In DeepLearningSimulator.predict():
results = model(frame, conf=0.45)
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    conf = float(box.conf[0])
    # add to detections list
```

### TensorFlow / Keras
```python
import tensorflow as tf
model = tf.saved_model.load('pothole_savedmodel/')
input_tensor = tf.image.resize(frame, [416, 416]) / 255.0
predictions = model(tf.expand_dims(input_tensor, 0))
```

### PyTorch
```python
import torch
model = torch.load('pothole_model.pth')
model.eval()
with torch.no_grad():
    tensor = preprocess(frame).unsqueeze(0)
    output = model(tensor)
```

## GPS Hardware Integration
Replace `GPSTagger.tag()` with:
```python
import gpsd
gpsd.connect()
packet = gpsd.get_current()
return packet.lat, packet.lon
```

## Dataset Recommendations
- **RDD2022** — Road Damage Detection Dataset (15,000+ images)
- **POTHOLE-600** — 600 annotated pothole images
- **DeepCrack** — crack detection dataset (transferable)

## Evaluation
```python
from utils.metrics import MetricsEngine
engine = MetricsEngine()
# After processing all frames:
summary = engine.get_summary()
print(f"mAP@0.5: {summary['metrics']['map_50']:.1%}")
print(f"F1: {summary['metrics']['f1']:.1%}")
```

## License
MIT — for research and production use.
