import streamlit as st
import requests
import numpy as np
import cv2

st.set_page_config(page_title="Brain MRI Segmentation + Chatbot", layout="wide")
st.title("Brain MRI Metastasis Segmentation")

API_BASE_URL = "http://127.0.0.1:8000"

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Image Segmentation")
    uploaded_files = st.file_uploader(
        "Upload MRI Images",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image_bytes = uploaded_file.read()
            image_np = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)

            st.image(img, caption=f"Uploaded image: {uploaded_file.name}", channels="GRAY")

            files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
            try:
                response = requests.post(f"{API_BASE_URL}/predict/", files=files, timeout=60)
                response.raise_for_status()
                prediction = response.json()
                st.write(f"Prediction for {uploaded_file.name}:")
                st.json(prediction)
            except requests.RequestException as exc:
                st.error(f"Unable to get prediction for {uploaded_file.name}. Details: {exc}")

with col2:
    st.subheader("MRI Assistant Chatbot")
    st.caption("Ask questions about file formats, predictions, and troubleshooting.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! I can help you use the MRI segmentation app. What do you want to know?",
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Type your question...")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        try:
            response = requests.post(
                f"{API_BASE_URL}/chat/",
                json={"message": user_prompt},
                timeout=30,
            )
            response.raise_for_status()
            bot_reply = response.json().get("reply", "Sorry, I could not generate a response.")
        except requests.RequestException as exc:
            bot_reply = f"Chat service is unavailable right now: {exc}"

        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
