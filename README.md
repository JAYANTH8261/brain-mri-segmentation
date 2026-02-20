# Brain MRI Segmentation with Chatbot

This project provides:
- A **FastAPI backend** for MRI segmentation inference.
- A **Streamlit frontend** for image upload and interactive results.
- A built-in **chatbot endpoint** (`/chat/`) for user guidance and troubleshooting.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.api:app --reload
```

In another terminal:

```bash
streamlit run app/streamlit_app.py
```

## API endpoints

- `POST /predict/`: Upload a single MRI image (`png`, `jpg`, `jpeg`, `tif`, `tiff`) and get prediction output.
- `POST /chat/`: Send JSON like `{ "message": "What file formats are supported?" }` and receive a chatbot response.
- `GET /`: Health hello endpoint.
