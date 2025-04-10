from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw
from mtcnn import MTCNN
import pickle
import io
import cv2
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Setting Up FastAPI & CORS (cross-origin resource sharing which allow frontend and backend to communicate)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
skintone_model = load_model("models/vgg1603_skintone_model.keras")
with open("models/undertone_model_optimized.pkl", "rb") as f:
    undertone_model = pickle.load(f)

# Initialize Face Detector
detector = MTCNN()

# Preprocessing Functions
#Skin Tone
def load_and_preprocess_image(image: Image.Image):
    img_array = np.array(image)
    faces = detector.detect_faces(img_array)

    if not faces:
        print("No face detected. Using full image.")
        cropped_image = image
    else:
        x, y, w, h = faces[0]["box"]
        x, y = abs(x), abs(y)
        cropped_image = image.crop((x, y, x + w, y + h))

    cropped_image = cropped_image.resize((224, 224))
    img_array = np.array(cropped_image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Undertone
def preprocess_image_undertone(image: Image.Image):
    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    return img_array.flatten().reshape(1, -1)


# Prediction Functions
def predict_skintone(image: Image.Image):
    img_array = load_and_preprocess_image(image)
    prediction = skintone_model.predict(img_array)[0][0]

    if prediction < 1:
        return "fair"
    elif prediction < 2:
        return "light"
    elif prediction < 3:
        return "light-medium"
    elif prediction < 4:
        return "medium"
    elif prediction < 5:
        return "medium-deep"
    else:
        return "deep"

def predict_undertone(image: Image.Image):
    features = preprocess_image_undertone(image)
    undertone_prob = list(undertone_model.predict_proba(features)[0])
    undertone_labels = ["warm", "cool", "neutral"]
    max_index = np.argmax(undertone_prob)
    return {"undertone": undertone_labels[max_index], "scores": undertone_prob}

# API Endpoints
@app.get("/")
def home():
    return {"message": "Skin Tone & Undertone Prediction API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        skintone_result = predict_skintone(image)
        undertone_result = predict_undertone(image)

        return JSONResponse(
            content={
                "predicted_skin_tone": skintone_result, # since skintone return string
                "predicted_undertone": undertone_result["undertone"]
            }
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/capture/")
def capture_image():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return JSONResponse(content={"error": "Could not access webcam"}, status_code=500)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return JSONResponse(content={"error": "Failed to capture image"}, status_code=500)

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        skintone_result = predict_skintone(image)
        undertone_result = predict_undertone(image)

        return JSONResponse(
            content={
                "predicted_skin_tone": skintone_result,
                "predicted_undertone": undertone_result,
            }
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)