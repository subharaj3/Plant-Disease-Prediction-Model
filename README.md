# Plant Disease Diagnostic Platform

A robust, microservice-based web application that utilizes Deep Learning to diagnose plant diseases from leaf images.

## 🏗️ System Architecture

This project is decoupled into three distinct layers to ensure modularity and scalability:

1. **Client Interface (Frontend):** A responsive, browser-native application built with Vanilla JavaScript, HTML5, and CSS3. It features a custom drag-and-drop file upload system and asynchronous API integration.
2. **API Gateway (Backend - Node.js):** The core routing server built with **Node.js, Express, and TypeScript**. It acts as the intermediary, securely handling `multipart/form-data` uploads from the client and routing them to the internal ML microservice.
3. **Inference Microservice (Backend - Python):** A dedicated **FastAPI** service running a custom 5-layer Convolutional Neural Network (CNN) built with **PyTorch**. It receives image buffers, processes the tensor transformations, and returns the classification and confidence score.

## 🚀 Tech Stack

- **Gateway Server:** Node.js, Express, TypeScript, Multer, Axios
- **Machine Learning Service:** Python, FastAPI, PyTorch, Torchvision, Uvicorn
- **Frontend:** Vanilla JavaScript, HTML, CSS (No external libraries)
- **Dataset:** PlantVillage (38 distinct crop-disease classes)

## ⚙️ How to Run Locally

### 1. Start the ML Inference Service

Navigate to the `backend-ml` directory and start the FastAPI server:

```bash
cd backend-ml
pip install fastapi uvicorn python-multipart torch torchvision Pillow
uvicorn main:app --reload
```
