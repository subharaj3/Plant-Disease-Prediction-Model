from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

app = FastAPI(title="Plant Disease Prediction API")

# Allow CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 1. MODEL ARCHITECTURE (From your Notebook)
# ==========================================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            ## CNN Layer 1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ## CNN Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ## CNN Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ## CNN Layer 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ## CNN Layer 5
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(256, 38))

    def forward(self, x):
        x = self.conv(x)
        x = self.global_pool(x)
        x = self.fc1(x)
        return x


def get_model():
    model = CNN()

    # Load your saved weights
    # map_location='cpu' ensures it works on your machine even without a GPU
    state_dict = torch.load(
        "model.pth", map_location=torch.device("cpu"), weights_only=True
    )

    # Note: Because your notebook had "model = nn.DataParallel(model)",
    # your saved weights might have "module." prefixed to their names.
    # This block safely removes that prefix if it exists so it loads correctly.
    unwrapped_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        unwrapped_state_dict[new_key] = value

    model.load_state_dict(unwrapped_state_dict)
    model.eval()  # Set to evaluation mode
    return model


model = get_model()

# ==========================================
# 2. CLASS NAMES (Standard 38 PlantVillage Classes)
# ==========================================
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Bacterial_spot",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ==========================================
# 3. IMAGE PREPROCESSING (From your validation transform)
# ==========================================
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def predict_image(image_bytes):
    try:
        # Open image and ensure it's in RGB format
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Apply transforms and add batch dimension (C, H, W) -> (1, C, H, W)
        tensor = transform(image).unsqueeze(0)

        # Disable gradient calculation for faster inference
        with torch.no_grad():
            outputs = model(tensor)
            # Get the predicted class index
            _, predicted_idx = torch.max(outputs, 1)

            # Get confidence score using Softmax
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence = probabilities[predicted_idx[0]].item()

        return {
            "class": CLASS_NAMES[predicted_idx.item()],
            "confidence": round(confidence * 100, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 4. THE API ENDPOINT
# ==========================================
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    # Read image bytes
    image_bytes = await file.read()

    # Get prediction
    result = predict_image(image_bytes)

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
