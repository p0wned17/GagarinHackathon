import re

import cv2
import numpy as np
import torch

# Constants for document classification
ID_TO_CLASSNAME = {
    0: {"type": "personal_passport", "page_number": "1"},
    1: {"type": "personal_passport", "page_number": "2"},
    2: {"type": "vehicle_passport", "page_number": None},
    3: {"type": "driver_license", "page_number": "1"},
    4: {"type": "driver_license", "page_number": "2"},
    5: {"type": "vehicle_certificate", "page_number": "1"},
    6: {"type": "vehicle_certificate", "page_number": "2"},
}


def preprocess(image: np.ndarray) -> torch.Tensor:
    """Preprocess the image for model inference."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    image = (image - mean) / std
    return torch.from_numpy(image).unsqueeze(0)


def remove_non_alphanumeric(text: str) -> str:
    """Remove non-alphanumeric characters from text."""
    return re.sub(r"[^\w]", "", text)
