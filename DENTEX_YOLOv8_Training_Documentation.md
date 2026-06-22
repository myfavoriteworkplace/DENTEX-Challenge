# DENTEX YOLOv8 Dental X-Ray Detection — Complete Training Documentation

End-to-end technical guide covering: Google Colab setup → DENTEX dataset discovery → COCO annotation conversion → YOLO folder creation → training → locating `best.pt` → validation → inference.

---

## Table of Contents

1. [Project Objective](#1-project-objective)
2. [Environment Setup](#2-environment-setup)
3. [Explore DENTEX Repository](#3-explore-dentex-repository)
4. [Download DENTEX Dataset](#4-download-dentex-dataset)
5. [Dataset Loading](#5-dataset-loading)
6. [Extract Training Data](#6-extract-training-data)
7. [Select Correct Annotation Dataset](#7-select-correct-annotation-dataset)
8. [Load Training JSON](#8-load-training-json)
9. [Inspect Annotation Format](#9-inspect-annotation-format)
10. [Understand Categories](#10-understand-categories)
11. [YOLO Dataset Requirement](#11-yolo-dataset-requirement)
12. [Create YOLO Structure](#12-create-yolo-structure)
13. [Convert COCO Bounding Boxes to YOLO](#13-convert-coco-bounding-boxes-to-yolo)
14. [Generate Training Labels](#14-generate-training-labels)
15. [Validation Dataset Conversion](#15-validation-dataset-conversion)
16. [Final Dataset Structure](#16-final-dataset-structure)
17. [Create YOLO Configuration](#17-create-yolo-configuration)
18. [Install YOLO](#18-install-yolo)
18a. [Verify Dataset Classes Before Training](#18a-verify-dataset-classes-before-training)
19. [Train YOLOv8](#19-train-yolov8)
20. [Training Output](#20-training-output)
21. [Locate Model](#21-locate-model)
22. [Load Trained Model](#22-load-trained-model)
23. [Validate Model](#23-validate-model)
24. [Run Prediction](#24-run-prediction)
25. [View Result](#25-view-result)
25a. [Model Verification Before Deployment](#25a-model-verification-before-deployment)
26. [Final Achievement](#final-achievement)
27. [Current Status & Next Steps](#current-status--next-steps)

---

## 1. Project Objective

### Goal

Build a custom dental X-ray AI detection model using:

- DENTEX Challenge Dataset
- YOLOv8 Object Detection
- Ultralytics framework
- Google Colab training environment

### Final Output

A trained YOLO model (`best.pt`) capable of detecting dental findings from X-ray images.

---

## 2. Environment Setup

### Platform

**Google Colab**

Reason: free GPU availability, easy Python environment, suitable for YOLO training.

### Step 1 — Install Required Libraries

```python
!pip install ultralytics datasets
```

### Purpose

**Ultralytics** — provides YOLOv8 model, training pipeline, and prediction pipeline.

**HuggingFace datasets** — used to download DENTEX dataset and parse annotations.

### Step 2 — Verify Installation

```python
import ultralytics
ultralytics.checks()
```

Expected output:

```
Ultralytics YOLO installed successfully
```

---

## 3. Explore DENTEX Repository

### Clone Challenge Repository

```python
!git clone https://github.com/myfavoritenetworkplace/DENTEX-Challenge.git
%cd /content/DENTEX-Challenge
```

### Check Repository

```python
!ls
```

Example output:

```
docker
scripts
configs
README
```

### Observation

Repository contained: evaluation scripts, configs, and Docker setup.

**Missing**: `training_data.zip`, `validation_data.zip`

**Conclusion**: dataset must be downloaded separately.

---

## 4. Download DENTEX Dataset

Dataset source: HuggingFace — `ibrahimhamamci/DENTEX`

### Load Dataset

Initial attempt:

```python
from datasets import load_dataset
load_dataset("ibrahimhamamci/DENTEX")
```

**Problem**: `Couldn't infer same data file format`

**Reason**: dataset contains mixed formats:

| Split       | Format      |
|-------------|-------------|
| `train`     | imagefolder |
| `test`      | imagefolder |
| `validation`| json        |

**Solution**: load each split separately.

---

## 5. Dataset Loading

### Find Cache

```python
# Dataset is cached at:
# /root/.cache/huggingface/hub/datasets--ibrahimhamamci--DENTEX
```

### Structure

```
DENTEX/
    training_data.zip
    validation_data.zip
    test_data.zip
    validation_triple.json
```

---

## 6. Extract Training Data

Training structure:

```
training_data/
    quadrant/
    quadrant_enumeration/
    quadrant-enumeration-disease/
        train_quadrant_enumeration_disease.json
        xrays/
            train_111.png
            train_595.png
```

---

## 7. Select Correct Annotation Dataset

DENTEX provides multiple annotation levels:

| Level                          | Contains                                    |
|--------------------------------|---------------------------------------------|
| `quadrant`                     | Quadrant information only                   |
| `quadrant_enumeration`         | Tooth enumeration                           |
| `quadrant-enumeration-disease` | Bounding boxes **and** disease labels       |

**Selected**: `quadrant-enumeration-disease` — because YOLO requires object detection annotations.

---

## 8. Load Training JSON

```python
from datasets import load_dataset

train_json = "/content/dentex_train/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json"

train_ann = load_dataset(
    "json",
    data_files=train_json
)
```

---

## 9. Inspect Annotation Format

```python
sample = train_ann["train"][0]
print(sample.keys())
```

Output:

```
images
annotations
categories_1
categories_2
categories_3
```

---

## 10. Understand Categories

```python
sample["categories_3"]
```

Output:

```json
[
  {"id": 0, "name": "Impacted"},
  {"id": 1, "name": "Caries"},
  {"id": 2, "name": "Periapical Lesion"},
  {"id": 3, "name": "Deep Caries"}
]
```

Classes:

| ID | Name                 |
|----|----------------------|
| 0  | Impacted             |
| 1  | Caries               |
| 2  | Periapical Lesion    |
| 3  | Deep Caries          |

### Category Levels in DENTEX

| Level  | Meaning                     | category_id |
|--------|-----------------------------|--------------|
| 1      | Quadrant classification     | category_id_1 |
| 2      | Tooth enumeration           | category_id_2 |
| 3      | **Dental disease findings** | **category_id_3** |

For AI diagnosis assistance, `category_id_3` is selected because it represents actual dental findings.

---

## 11. YOLO Dataset Requirement

YOLO expects the following folder structure:

```
dataset/
  images/
    train/
      image.jpg
    val/
      image.jpg
  labels/
    train/
      image.txt
    val/
      image.txt
```

### Label Format

```
class x_center y_center width height
```

Example:

```
2 0.45 0.32 0.12 0.18
```

---

## 12. Create YOLO Structure

```python
import os

os.makedirs("/content/dentex_yolo/images/train", exist_ok=True)
os.makedirs("/content/dentex_yolo/images/val", exist_ok=True)
os.makedirs("/content/dentex_yolo/labels/train", exist_ok=True)
os.makedirs("/content/dentex_yolo/labels/val", exist_ok=True)
```

---

## 13. Convert COCO Bounding Boxes to YOLO

DENTEX format: `[x, y, width, height]`

YOLO needs: `[x_center, y_center, width, height]` (normalized)

```python
def coco_to_yolo(box, w, h):
    x, y, bw, bh = box
    return (
        (x + bw / 2) / w,
        (y + bh / 2) / h,
        bw / w,
        bh / h,
    )
```

---

## 14. Generate Training Labels

Process for every image:

1. Copy image to `images/train/`
2. Read its annotations from the COCO JSON
3. Extract disease class from each annotation:

```python
for annotation in image_annotations:
    cls = annotation["category_id_3"]  # Disease class: 0=Impacted, 1=Caries, 2=Periapical Lesion, 3=Deep Caries
    bbox = annotation["bbox"]  # COCO format
    yolo_bbox = coco_to_yolo(bbox, image_width, image_height)
```

4. Convert each bounding box using `coco_to_yolo`
5. Write `.txt` label file to `labels/train/` in format: `class x_center y_center width height`

**Output**:

```
images/train/
  train_111.jpg

labels/train/
  train_111.txt
```

**Example label file content**:

```
1 0.45 0.32 0.12 0.18
2 0.65 0.55 0.15 0.20
```

---

## 15. Validation Dataset Conversion

Validation JSON: `validation_triple.json`

Structure: `images`, `annotations`, `categories_1`, `categories_2`, `categories_3`

Converted the same way as training data.

---

## 16. Final Dataset Structure

```
dentex_yolo/
  images/
    train/
      1.jpg
    val/
      1.jpg
  labels/
    train/
      1.txt
    val/
      1.txt
```

---

## 17. Create YOLO Configuration

Create `dentex.yaml`:

```yaml
train: /content/drive/MyDrive/dentex_project/yolo/images/train
val: /content/drive/MyDrive/dentex_project/yolo/images/val

nc: 4

names:
  0: Impacted
  1: Caries
  2: Periapical Lesion
  3: Deep Caries
```

---

## 18. Install YOLO

```python
!pip install ultralytics
```

---

## 18a. Verify Dataset Classes Before Training

**Important**: Ensure your YOLO model will use meaningful disease labels.

```python
from ultralytics import YOLO

# Load base model to verify class configuration
model = YOLO("yolov8s.pt")
print("Base model class names:")
print(model.names)
```

After training completes, verify the trained model has correct labels:

```python
from ultralytics import YOLO
model = YOLO("/content/drive/MyDrive/dentex_project/yolo/runs/detect/train/weights/best.pt")
print("Trained model class names:")
print(model.names)
```

**Expected output**:

```
{0: 'Impacted', 1: 'Caries', 2: 'Periapical Lesion', 3: 'Deep Caries'}
```

If class names appear as `0`, `1`, `2`, `3` or `1`, `2`, `3`, `4`, the model was trained on wrong category level. Retrain using the corrected `dentex.yaml` file.

---

## 19. Train YOLOv8

Model: **YOLOv8s**

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="/content/drive/MyDrive/dentex_project/dentex.yaml",
    epochs=100,
    imgsz=1024,
    batch=8
)
```

---

## 20. Training Output

YOLO creates:

```
runs/
  detect/
    train/
      weights/
        best.pt
        last.pt
```

---

## 21. Locate Model

```python
!find / -name "best.pt"
```

Actual result:

```
/content/dentex_yolo/runs/detect/train/weights/best.pt
```

---

## 22. Load Trained Model

```python
from ultralytics import YOLO

model = YOLO(
    "/content/dentex_yolo/runs/detect/train/weights/best.pt"
)
```

---

## 23. Validate Model

```python
metrics = model.val()
```

Generates:

- Precision
- Recall
- mAP50
- mAP50-95

---

## 24. Run Prediction

```python
results = model.predict(
    source="/content/dentex_yolo/images/val/1.jpg",
    conf=0.25,
    save=True,
)
```

---

## 25. View Result

```python
results[0].show()
```

Output: dental X-ray with bounding boxes, confidence scores, and class IDs.

---

## 25a. Model Verification Before Deployment

Before integrating `best.pt` into your FastAPI backend or Hugging Face, verify the model outputs correct disease classes:

```python
from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/dentex_project/yolo/runs/detect/train/weights/best.pt")
print("Model class names:")
print(model.names)
```

**Expected**:

```
{0: 'Impacted', 1: 'Caries', 2: 'Periapical Lesion', 3: 'Deep Caries'}
```

**If you see quadrant names** (`1`, `2`, `3`, `4`), the model was trained on wrong category. Delete training results and retrain.

Only after this verification should `best.pt` be:
- Moved to Hugging Face Model Hub
- Deployed to FastAPI backend
- Used in BookMySlot AI diagnosis system

---

## Final Achievement

```
DENTEX Dataset (quadrant-enumeration-disease)
       |
       v
COCO Annotation Processing (using categories_3)
       |
       v
YOLO Dataset Conversion (disease classes)
       |
       v
YOLOv8 Training
       |
       v
Custom Dental Finding Detection Model
       |
       v
best.pt with meaningful disease labels
       |
       v
Ready for API Integration
```

### Classes Detected

- **Impacted teeth** (dental impaction)
- **Caries** (dental decay)
- **Periapical Lesion** (root infection indicator)
- **Deep Caries** (advanced tooth decay)

This model serves as the core AI engine for BookMySlot's diagnosis assistance system.

---

## Current Status & Next Steps

### Completed

- [x] Dataset preparation
- [x] Annotation conversion
- [x] YOLO training
- [x] Model creation
- [x] Inference testing

### Next Phase

```
best.pt
    |
    v
Backend API
    |
    v
React Application
    |
    v
Doctor Dashboard AI Assistance
```

---

> This is the complete ML pipeline documentation from zero to trained model.
