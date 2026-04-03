import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.inference.predictor import Predictor

app = FastAPI(
    title="MedMNIST Classification API",
    description="Serve predictions for a trained MedMNIST image classification model.",
    version="0.1.0",
)

predictor = Predictor(
    config_path="configs/config.yaml",
)


@app.get("/health")
def health():
    """
    Return the health status of the serving API.

    Returns
    -------
    dict
        Dictionary containing API status information, including the
        loaded checkpoint path and inference device.
    """
    return {
        "status": "ok",
        "checkpoint_path": str(predictor.checkpoint_path),
        "device": str(predictor.device),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Run prediction on an uploaded image.

    Parameters
    ----------
    file : fastapi.UploadFile
        Uploaded image file.

    Returns
    -------
    dict
        Prediction result returned by the underlying predictor.

    Raises
    ------
    HTTPException
        Raised if the uploaded file is not an image or if prediction fails.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = predictor.predict(image)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
