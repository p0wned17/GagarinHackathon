from fastapi import FastAPI, File, HTTPException, UploadFile
from image_processing import predict_image

app = FastAPI()


@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    image_data = await file.read()
    doc_type, page_number, confidence, series, number = predict_image(image_data)
    return {
        "type": doc_type,
        "page_number": page_number,
        "confidence": confidence,
        "series": series,
        "number": number,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=3333, log_level="info", workers=2)
