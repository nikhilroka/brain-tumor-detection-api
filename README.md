# Brain Tumor Detection Using Deep Learning

A complete deep learning and deployment pipeline for classifying brain MRI images as **Brain Tumor** or **Normal** using a pretrained **ResNet50** convolutional neural network.

The project demonstrates the complete workflow from model training and evaluation to API-based inference and Docker containerization.

> **Note:** This project is an educational/research prototype and is not intended for clinical diagnosis.

---

## 🚀 Project Overview

This project uses **transfer learning with ResNet50** to classify brain MRI images into two classes:

* 🧠 Brain Tumor
* ✅ Normal

The trained PyTorch model is exposed through a **FastAPI REST API** and packaged with **Docker**, allowing the application to run in a reproducible environment.

### Technology Pipeline

```text
MRI Image
    ↓
Image Preprocessing
    ↓
ResNet50
    ↓
Softmax Probability
    ↓
Brain Tumor / Normal
    ↓
Confidence Score
    ↓
FastAPI REST API
    ↓
Docker Container
```

---

## ✨ Key Features

* ResNet50 transfer learning
* PyTorch-based inference
* MRI image preprocessing
* Binary image classification
* Confidence score generation
* Test-set evaluation
* Accuracy, precision, recall and F1-score
* Confusion matrix
* FastAPI REST API
* Interactive Swagger API documentation
* Input file validation
* 10 MB upload size limit
* Docker containerization
* CPU-based inference support

---

## 🧠 Model

### Architecture

**ResNet50**

The model uses a pretrained ResNet50 architecture and replaces the original classification layer with a custom two-class output layer.

```text
ResNet50
    ↓
Feature Extraction
    ↓
Fully Connected Layer
    ↓
2 Classes
    ↓
Brain Tumor / Normal
```

### Transfer Learning

The pretrained ResNet50 architecture provides learned visual features that can be adapted to the brain MRI classification task.

Input image size:

```text
224 × 224 × 3
```

Output:

```text
2 classes
```

Classes:

```text
0 → Brain Tumor
1 → Normal
```

---

## 📊 Model Evaluation

The model was evaluated on a separate test set containing:

```text
Total test images: 53
```

### Test Accuracy

**81.13%**

### Classification Report

| Class                |  Precision |     Recall |   F1-Score | Support |
| -------------------- | ---------: | ---------: | ---------: | ------: |
| Brain Tumor          |     86.96% |     74.07% |     80.00% |      27 |
| Normal               |     76.67% |     88.46% |     82.14% |      26 |
| **Macro Average**    | **81.81%** | **81.27%** | **81.07%** |  **53** |
| **Weighted Average** | **81.91%** | **81.13%** | **81.05%** |  **53** |

### Confusion Matrix

```text
                  Predicted
               Brain Tumor   Normal

Actual
Brain Tumor         20          7
Normal               3         23
```

The model correctly classified:

* 20 Brain Tumor images
* 23 Normal images

Total correct predictions:

```text
43 / 53
```

---

## ⚙️ Image Preprocessing

Images are processed using the following pipeline:

```text
Resize
  ↓
Center Crop
  ↓
Convert to Tensor
  ↓
ImageNet Normalization
```

The inference preprocessing uses:

```text
Resize: 256
Center Crop: 224 × 224
Normalization:
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

---

## 🌐 FastAPI

The trained model is exposed through a REST API using FastAPI.

### Start the API locally

From the `api` directory:

```powershell
python -m uvicorn main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

### Swagger Documentation

Open:

```text
http://127.0.0.1:8000/docs
```

Swagger provides an interactive interface for testing the API.

---

## 🔌 API Endpoints

### GET `/`

Checks whether the API is running.

Example response:

```json
{
  "message": "Brain Tumor Detection API is running",
  "status": "success"
}
```

### GET `/health`

Returns the API and model health status.

Example:

```json
{
  "status": "healthy",
  "model": "ResNet50",
  "device": "CPU"
}
```

### POST `/predict`

Accepts an MRI image and returns the predicted class and confidence.

Supported formats:

```text
JPG
JPEG
PNG
```

Maximum file size:

```text
10 MB
```

Example response:

```json
{
  "status": "success",
  "filename": "Y185.jpg",
  "prediction": "Brain Tumor",
  "confidence": 0.6528,
  "confidence_percent": 65.28,
  "model": "ResNet50",
  "model_version": "1.0"
}
```

---

## 🐳 Docker Deployment

The application is containerized using Docker.

### Build the Docker image

Run this command from the project root:

```powershell
docker build -t brain-tumor-api .
```

### Run the container

```powershell
docker run --name brain-tumor-container -p 8000:8000 brain-tumor-api
```

The API will then be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### Docker Architecture

```text
Docker Container
│
├── Python 3.11
├── FastAPI
├── Uvicorn
├── PyTorch
├── Torchvision
├── Pillow
├── API
│   └── main.py
│
└── Model
    └── brain_tumor_resnet50.pth
```

---

## 📁 Project Structure

```text
Brain Tumor Project/
│
├── api/
│   ├── main.py
│   └── requirements.txt
│
├── models/
│   └── brain_tumor_resnet50.pth
│
├── Dataset/
│   ├── Train/
│   │   ├── Brain Tumor/
│   │   └── Normal/
│   │
│   └── Test/
│       ├── Brain Tumor/
│       └── Normal/
│
├── results/
│
├── frontend/
│
├── Dockerfile
├── .dockerignore
├── README.md
│
└── Brain Tumor Detection Project.ipynb
```

---

## 🛠️ Technologies Used

### Programming

* Python

### Deep Learning

* PyTorch
* Torchvision
* ResNet50
* Transfer Learning

### Computer Vision

* PIL
* Image preprocessing
* Image classification

### API Development

* FastAPI
* Uvicorn
* REST API
* Swagger/OpenAPI

### Deployment

* Docker
* Dockerfile
* Containerized inference

### Data Science

* NumPy
* Scikit-learn
* Matplotlib

---

## 🔬 Development Workflow

```text
1. Dataset Preparation
        ↓
2. Image Preprocessing
        ↓
3. ResNet50 Transfer Learning
        ↓
4. Model Training
        ↓
5. Model Evaluation
        ↓
6. Save Trained Model
        ↓
7. FastAPI Integration
        ↓
8. Local API Testing
        ↓
9. Dockerization
        ↓
10. Container Testing
```

---

## ⚠️ Limitations

This project has several limitations:

* The test dataset contains only 53 images.
* The reported performance should not be interpreted as clinical performance.
* The model has not undergone clinical validation.
* MRI acquisition protocols may differ between hospitals and datasets.
* Dataset size and diversity can affect generalization.
* Confidence scores represent model probabilities and should not be interpreted as certainty.

---

## 🔮 Future Improvements

Potential improvements include:

* Larger and more diverse MRI datasets
* Cross-validation
* Hyperparameter optimization
* Data augmentation improvements
* Class imbalance analysis
* Grad-CAM visual explanations
* Model explainability
* Automated evaluation reports
* Frontend web application
* Cloud deployment
* CI/CD pipeline
* Model versioning
* Monitoring and logging
* GPU inference optimization

---

## 💼 Portfolio / Engineering Value

This project demonstrates practical skills across the machine learning lifecycle:

```text
Machine Learning
      +
Deep Learning
      +
Computer Vision
      +
Model Evaluation
      +
REST API Development
      +
Docker
      +
Deployment
```

It demonstrates how a trained deep learning model can be transformed from a research notebook into a **usable inference service**.

---

## 📌 Disclaimer

This project is intended for **educational, research, and software engineering demonstration purposes only**.

It should not be used as a medical diagnostic system or as a substitute for professional medical evaluation.
