---
license: mit
tags:
  - YOLOv8
  - Object Detection
  - Computer Vision
  - License Plate Detection
  - Ultralytics
  - Real-time Detection
library_name: ultralytics
inference: false
---

# 🚗 YOLOv8 License Plate Detection Model

This repository contains a YOLOv8 object detection model trained to detect **license plates** in real-world images. The model was trained using the [Ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8 framework and can be deployed for real-time applications such as surveillance, traffic monitoring, and vehicle identification.

---

## 🧠 Model Details

- **Architecture**: YOLOv8n (Nano variant)
- **Framework**: [Ultralytics YOLOv8](https://docs.ultralytics.com)
- **Task**: Object Detection
- **Classes**: 1 (`license_plate`)
- **Input resolution**: 640×640
- **File**: `best.pt`

---

## 🔧 How to Use

Install dependencies first:

```bash
pip install ultralytics
```

## Example usage:

```python
from ultralytics import YOLO

# Load model from HF
model = YOLO("koushik-ai/yolov8-license-plate-detection/best.pt")

# Run inference
results = model("your_image.jpg")

# Show results
results[0].show()
```