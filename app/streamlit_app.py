import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image  # Ensure you import this for image handling

st.title("Brain MRI Metastasis Segmentation")

# File uploader allowing multiple file uploads
uploaded_files = st.file_uploader("Upload MRI Images", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Read the uploaded image
        image = uploaded_file.read()
        
        # Convert the bytes to a NumPy array and then to an OpenCV image
        image_np = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)
        
        # Optional: Display the uploaded image (for confirmation)
        st.image(img, channels="GRAY")

        # Send a request to the FastAPI server for prediction
        response = requests.post("http://127.0.0.1:8000/predict/", files={"file": uploaded_file})
        
        if response.status_code == 200:
            prediction = response.json()
            st.write(f"Prediction for {uploaded_file.name}: {prediction}")
        else:
            st.error(f"Error: Unable to get prediction for {uploaded_file.name}.")
