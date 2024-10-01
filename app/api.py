# app/api.py

from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = FastAPI()

model = load_model('models/model_weights/best_model.h5')

@app.post("/predict/")
async def predict_mri(file: UploadFile = File(...)):
    if not file.filename.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        return {"error": "File format not allowed. Please upload a PNG, JPG, or TIF file."}
    image = await file.read()
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256)) / 255.0
    image = np.expand_dims(image, axis=[0, -1])
    
    prediction = model.predict(image)
    return {"prediction": prediction.tolist()}

@app.get("/")
def read_root():
    return {"Hello": "World"}