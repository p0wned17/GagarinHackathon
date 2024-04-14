import re

import cv2
import numpy as np
import torch
from model_utils import (
    orient_model,
    text_detection_model,
    text_recognition_model,
    type_model,
)
from utils import ID_TO_CLASSNAME


def predict_type(image: torch.Tensor) -> tuple:
    """Predict the document type and page number."""
    with torch.inference_mode():
        scores = torch.softmax(type_model(image.to("cuda:0").half()), dim=1).cpu()
        predicted_idx = torch.argmax(scores, dim=1).item()
        class_info = ID_TO_CLASSNAME[predicted_idx]
        return (
            class_info["type"],
            class_info["page_number"],
            scores[0][predicted_idx].item(),
        )


def predict_orient(image: torch.Tensor) -> int:
    """Predict the orientation of the document."""
    with torch.inference_mode():
        scores = torch.softmax(orient_model(image.to("cuda:0").half()), dim=1).cpu()
        return torch.argmax(scores, dim=1).item()


def recognize_text(image: np.ndarray, remove_letters: bool = False) -> tuple:
    """Recognize text from the image using OCR."""
    result = text_recognition_model.ocr(image)
    text = "".join([line[1][0] for line in result[0]])
    text = re.sub(r"[^\w]", "", text)
    if remove_letters:
        text = re.sub(r"[a-zA-Z]", "", text)
    if len(text) < 4 or len(text) > 20:
        return "unknown", "unknown"
    return text[:4], text[4:]


def detect_text(image: np.ndarray) -> np.ndarray:
    """Detect text area in the image using YOLO model."""
    results = text_detection_model.predict(image, max_det=1, conf=0.5)
    if len(results) == 0:
        return None
    bbox = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
    padding = 10
    x1, y1, x2, y2 = (
        max(0, bbox[0] - padding),
        max(0, bbox[1] - padding),
        min(image.shape[1], bbox[2] + padding),
        min(image.shape[0], bbox[3] + padding),
    )
    return image[y1:y2, x1:x2]


def rotate_image_based_on_orientation(
    image: np.ndarray, orient_key: int, doc_type: str
) -> np.ndarray:
    """Rotate image based on predicted orientation."""
    rotations = (
        {
            0: cv2.ROTATE_90_COUNTERCLOCKWISE,
            1: cv2.ROTATE_180,
            2: cv2.ROTATE_90_CLOCKWISE,
            3: None,
        }
        if doc_type == "personal_passport"
        else {
            0: None,
            1: cv2.ROTATE_90_COUNTERCLOCKWISE,
            2: cv2.ROTATE_180,
            3: cv2.ROTATE_90_CLOCKWISE,
        }
    )
    if rotations[orient_key] is not None:
        return cv2.rotate(image, rotations[orient_key])
    return image
