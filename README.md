# Plant-Disease-Prediction-Model

# 🌿 Plant Disease Prediction using CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Accuracy](https://img.shields.io/badge/Accuracy-98.7%25-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview
This project implements a Convolutional Neural Network (CNN) to detect plant diseases from images of leaves. The model is trained on the **New Plant Diseases Dataset** and is capable of classifying images into **38 distinct classes** (including healthy and diseased states) with a high degree of accuracy.

The project focuses on robust model training techniques to handle overfitting, utilizing data augmentation and an optimized CNN architecture.

## 📂 Dataset
The model was trained using the **New Plant Diseases Dataset (Augmented)**, sourced via KaggleHub.
- **Total Classes:** 38
- **Total Images:** ~87,000 RGB images
- **Dataset Link:** [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## 🏗️ Model Architecture
The custom CNN architecture is designed for efficiency and generalization:
1.  **Input Layer:** 256x256 RGB Images.
2.  **Convolutional Blocks:** 5 blocks, each consisting of:
    -   `Conv2d` (Feature Extraction)
    -   `BatchNorm2d` (Stabilization)
    -   `ReLU` (Activation)
    -   `MaxPool2d` (Dimensionality Reduction)
3.  **Global Average Pooling:** Replaces traditional `Flatten` layers to minimize parameter count and reduce overfitting.
4.  **Fully Connected Layer:** A final dense layer maps features to the 38 output classes.

## 🚀 Key Features & Improvements
The initial model suffered from severe overfitting (100% Train / 32% Test). The following techniques were implemented to achieve **98.7% Test Accuracy**:

* **Data Augmentation:** Applied random rotations, horizontal flips, and color jitter during training to force the model to learn features rather than memorizing pixels.
* **Adaptive Pooling:** Switched to `AdaptiveAvgPool2d` to drastically reduce the number of trainable parameters.
* **Corrected Training Loop:** Implemented real-time batch metrics to monitor training vs. validation loss accurately.
* **Early Stopping:** (Optional) Mechanism to stop training when validation accuracy plateaus.

## 📊 Results
| Metric | Value |
| :--- | :--- |
| **Training Accuracy** | ~99% |
| **Validation Accuracy** | **98.7%** |
| **Loss Function** | CrossEntropyLoss |
| **Optimizer** | AdamW (lr=0.001) |

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.x
* PyTorch
* Torchvision
* Matplotlib & Seaborn (for visualization)

### Running the Notebook
1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/plant-disease-prediction.git](https://github.com/your-username/plant-disease-prediction.git)
    ```
2.  Install dependencies:
    ```bash
    pip install torch torchvision matplotlib seaborn scikit-learn kagglehub
    ```
3.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook Plant_Disease_Prediction.ipynb
    ```

## 🔮 Future Scope
* **Mobile Deployment:** Converting the model to TFLite or ONNX for use in a mobile app.
* **Real-world Testing:** Fine-tuning the model on "wild" images (noisy backgrounds) to improve field performance.
* **Web Interface:** Building a Streamlit or Flask app to allow users to upload leaf photos for instant analysis.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License
This project is open-source and available under the MIT License.
