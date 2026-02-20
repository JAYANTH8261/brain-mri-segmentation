from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = FastAPI()

model = load_model("models/model_weights/best_model.h5")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


def _chatbot_reply(message: str) -> str:
    """Rule-based assistant for MRI segmentation workflow guidance."""
    normalized = message.lower().strip()

    if any(keyword in normalized for keyword in ["hello", "hi", "hey"]):
        return (
            "Hi! I can help you use this MRI segmentation tool. "
            "Ask me about supported file types, how predictions work, or how to troubleshoot errors."
        )

    if any(keyword in normalized for keyword in ["file", "format", "upload", "type"]):
        return (
            "You can upload PNG, JPG/JPEG, and TIF/TIFF MRI images. "
            "The API endpoint is /predict/ and accepts one image per request."
        )

    if any(keyword in normalized for keyword in ["predict", "segmentation", "result", "output"]):
        return (
            "For each uploaded MRI image, the model resizes it to 256x256 grayscale, "
            "normalizes pixel values, then returns a predicted segmentation mask tensor as JSON."
        )

    if any(keyword in normalized for keyword in ["error", "fail", "not working", "problem", "issue"]):
        return (
            "Common checks: ensure the FastAPI server is running at http://127.0.0.1:8000, "
            "upload only supported image formats, and confirm model weights exist at "
            "models/model_weights/best_model.h5."
        )

    if any(keyword in normalized for keyword in ["bye", "thanks", "thank you"]):
        return "You're welcome! If you want, I can help with API usage examples next."

    return (
        "I can help with MRI upload requirements, prediction behavior, and troubleshooting. "
        "Try asking: 'What image formats are supported?'"
    )


@app.post("/predict/")
async def predict_mri(file: UploadFile = File(...)):
    if not file.filename.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        return {"error": "File format not allowed. Please upload a PNG, JPG, or TIF file."}

    image = await file.read()
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256)) / 255.0
    image = np.expand_dims(image, axis=[0, -1])

    prediction = model.predict(image)
    return {"prediction": prediction.tolist()}


@app.post("/chat/", response_model=ChatResponse)
def chat(chat_request: ChatRequest):
    reply = _chatbot_reply(chat_request.message)
    return ChatResponse(reply=reply)


@app.get("/")
def read_root():
    return {"Hello": "World"}
