🚧 PotholeAI – Smart Pothole Detection System

This is an intelligent computer vision-based project designed to **detect and analyze potholes** from images and videos. It uses a hybrid approach combining **Classical Computer Vision** and **Deep Learning (YOLO-ready)**, along with a **Streamlit dashboard** for visualization.

🚀 Features :
  - Hybrid Detection: Combines Classical CV + Deep Learning (YOLO-ready)
  - Smart UI: Interactive multi-tab dashboard using Streamlit
  - Severity Analysis: Classifies potholes (Minor, Moderate, Severe, Critical)
  - GPS Tagging: Associates potholes with geographic coordinates
  - Duplicate Detection: Uses perceptual hashing (pHash)
  - Metrics Dashboard: Precision, Recall, F1-score, mAP evaluation
  - Synthetic Data Generator: No dataset required for demo/testing
  - Video Processing: Multi-frame pothole detection

🛠️ Tech Stack:
  - Backend: Python
  - Machine Learning: YOLO (extendable), NumPy
  - Computer Vision: OpenCV
  - Frontend/UI: Streamlit
  - Visualization: Plotly
  - Evaluation: Scikit-learn
  
📁 Project Structure:

<img width="557" height="280" alt="image" src="https://github.com/user-attachments/assets/4e9c99e2-c69d-4ea4-aeb4-869e96cc6d03" />



⚙️ Setup & Installation :
  - 1. Install Dependencies
    - pip install -r requirements.txt  
  - ▶️ Run the Application
      - streamlit run app.py
  - Then open in browser:
    - http://localhost:8501

🧪Testing & Demo :
  - No dataset required
  - Uses synthetic data generator
  - Supports:
    - Image upload
    - Camera input
    - Generated samples

🤖 Deep Learning Integration :

- To use a real YOLO model:
  1. Install:
  - pip install ultralytics
  
  2. Replace `DeepLearningSimulator.predict()` with:
  - from ultralytics import YOLO
  - model = YOLO("best.pt")
  - results = model(image)

📊 Sample Output:

- Detection: Bounding boxes around potholes
- Severity: Minor / Moderate / Severe / Critical
- Metrics:
  - Precision: 92%
  - Recall: 90.2%
  - mAP@0.5: 58.5%

🚀 Future Enhancements:

* Real-time mobile app integration
* Cloud deployment (web hosting)
* Live GPS tracking system
* Alert system for critical potholes
* Smart city dashboard integration

🌍 Applications:
    - Road safety monitoring
    - Smart city infrastructure
    - Municipal maintenance systems
    - Autonomous vehicle support
