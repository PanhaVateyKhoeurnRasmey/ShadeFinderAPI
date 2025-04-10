import sys
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import tempfile
import os
import logging
from utils.image_processing import UndertonePredictor
import utils.match_recommend
from utils.new_preprocess import load_and_preprocess_image, preprocess_image_undertone
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load V1 model
model_v1 = UndertonePredictor('models/skin_tone_classifier.pkl')

# Load V2 models
skintone_model_v2 = load_model('models/vgg1603_skintone_model.keras')
with open('models/undertone_model_optimized.pkl', "rb") as f:
    undertone_model_v2 = pickle.load(f)

# Health Check Endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# V1 Endpoint
@app.post("/v1/predict/")
async def predict_v1(file: UploadFile = File(...)):
    try:
        print("Starting prediction (V1):")
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Save the image to a temporary file for V1
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp.name)
            img_path = tmp.name

        try:
            # Perform predictions using the saved image path
            undertone = model_v1.predict_undertone(img_path)  # warm, cool, neutral
            tone_palette = ['#533023', '#6C4131', '#A36F48', '#BF8861', '#ECD0BA', '#F8E5D6']
            tone_labels = ['deep', 'medium-deep', 'medium', 'light-medium', 'light', 'fair']
            tone = model_v1.predict_tone(img_path, tone_palette, tone_labels)
        finally:
            # Ensure the temporary file is removed
            os.remove(img_path)
            img.close()

        recommendations = utils.match_recommend.get_recommendation(undertone["undertone"], tone["tone_label"])
        if recommendations is None:
            raise HTTPException(status_code=404, detail="No matching product found")

        response = {
            'undertone': undertone,
            'tone': tone,
            'recommendations': recommendations
        }
        print("Returning response:\n", response)
        return JSONResponse(content=response, status_code=200)

    except Exception as e:
        print("error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# V2 Endpoint
@app.post("/v2/predict/")
async def predict_v2(file: UploadFile = File(...)):
    try:
        print("Starting prediction (V2):")
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Save the image to a temporary file for V2
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp.name)
            img_path = tmp.name

        try:
            # Predict skintone for V2
            skintone_result = predict_skintone(img)
            undertone_result = predict_undertone(img)
        finally:
            # Ensure the temporary file is removed
            os.remove(img_path)
            img.close()

        # Get recommendations for V2
        recommendations = utils.match_recommend.get_recommendation(undertone_result["undertone"], skintone_result)

        if recommendations is None:
            raise HTTPException(status_code=404, detail="No matching product found")

        response = {
            'tone': {
                'tone_label': skintone_result
            },
            'undertone': {
                'undertone': undertone_result["undertone"]
            },
            'recommendations': recommendations
        }

        print("Returning response (V2):\n", response)
        return JSONResponse(content=response, status_code=200)

    except Exception as e:
        print("error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# Helper Functions for V2

def predict_skintone(image: Image.Image):
    img_array = load_and_preprocess_image(image)
    prediction = skintone_model_v2.predict(img_array)[0][0]
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
    undertone_prob = list(undertone_model_v2.predict_proba(features)[0])
    undertone_labels = ["warm", "cool", "neutral"]
    max_index = np.argmax(undertone_prob)
    return {"undertone": undertone_labels[max_index], "scores": undertone_prob}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
