from io import BytesIO
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from configs.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    IMAGE_EXTENSIONS,
    MODEL_PATH,
    OUTPUT_DIR,
    THRESHOLD,
)


def load_prediction_settings(settings_path=None):
    """Load the model and validation threshold selected by the notebook."""
    settings_path = Path(settings_path or OUTPUT_DIR / "selected_threshold.json")
    if not settings_path.is_file():
        raise FileNotFoundError(
            f"Selected model settings not found: {settings_path}. "
            "Train and evaluate a model before using inference."
        )

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    checkpoint = Path(settings["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = settings_path.parent.parent / checkpoint
    threshold = float(settings["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid classification threshold: {threshold}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected model checkpoint not found: {checkpoint}")
    return checkpoint, threshold


def preprocess_uploaded_image(image_bytes, model):
    """Decode and preprocess an upload according to the model input contract."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image_format = image.format
            image.verify()
    except (OSError, ValueError, UnidentifiedImageError) as error:
        raise ValueError("Upload must be a readable JPEG, PNG, or BMP image") from error

    if image_format not in {"JPEG", "PNG", "BMP"}:
        raise ValueError("Only JPEG, PNG, and BMP images are supported")

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        raise ValueError("Only single-input image models are supported")
    height = input_shape[1] or IMAGE_SIZE[0]
    width = input_shape[2] or IMAGE_SIZE[1]
    channels = input_shape[3]

    if channels not in (1, 3):
        raise ValueError(f"Expected a one- or three-channel model, got {channels}")

    if channels == 3:
        with Image.open(BytesIO(image_bytes)) as source_image:
            grayscale_image = source_image.convert("L").resize(
                (IMAGE_SIZE[1], IMAGE_SIZE[0])
            )
            image_array = np.asarray(grayscale_image, dtype=np.float32) / 255.0
        image = tf.convert_to_tensor(image_array[..., np.newaxis])
        image = tf.image.resize(image, (height, width))
        image = tf.image.grayscale_to_rgb(image)
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        image = preprocess_input(image * 255.0)
    else:
        image = tf.io.decode_image(
            image_bytes,
            channels=channels,
            expand_animations=False,
        )
        image.set_shape([None, None, channels])
        image = tf.image.resize(image, (height, width))
        image = tf.cast(image, tf.float32)
        image = image / 255.0

    return tf.expand_dims(image, axis=0)


def predict_uploaded_image(image_bytes, model, threshold):
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid classification threshold: {threshold}")

    image = preprocess_uploaded_image(image_bytes, model)
    probabilities = np.asarray(model.predict(image, verbose=0)).reshape(-1)
    if probabilities.size != 1:
        raise ValueError("Expected a model with one binary-classification output")

    pneumonia_probability = float(probabilities[0])
    if not 0.0 <= pneumonia_probability <= 1.0:
        raise ValueError(f"Model returned invalid probability: {pneumonia_probability}")
    class_index = int(pneumonia_probability >= threshold)
    return {
        "prediction": CLASS_NAMES[class_index],
        "probability_normal": 1.0 - pneumonia_probability,
        "probability_pneumonia": pneumonia_probability,
        "threshold": float(threshold),
    }


def predict_image(image_path, model_path=MODEL_PATH):
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    image_bytes = image_path.read_bytes()
    result = predict_uploaded_image(image_bytes, model, THRESHOLD)
    return {
        "image": str(image_path),
        **result,
        "pneumonia_probability": result["probability_pneumonia"],
    }
