from io import BytesIO
import json

import numpy as np
import pytest
from PIL import Image

from src.predict import (
    load_prediction_settings,
    predict_uploaded_image,
    preprocess_uploaded_image,
)


class DummyBinaryModel:
    input_shape = (None, 8, 8, 1)

    def predict(self, images, verbose=0):
        assert tuple(images.shape) == (1, 8, 8, 1)
        return np.array([[0.75]], dtype=np.float32)


class DummyRgbModel(DummyBinaryModel):
    input_shape = (None, 8, 8, 3)

    def predict(self, images, verbose=0):
        assert tuple(images.shape) == (1, 8, 8, 3)
        return np.array([[0.75]], dtype=np.float32)


def _png_bytes():
    output = BytesIO()
    Image.fromarray(np.full((12, 16), 128, dtype=np.uint8)).save(output, format="PNG")
    return output.getvalue()


def _color_png_bytes():
    output = BytesIO()
    Image.new("RGB", (12, 16), color=(255, 0, 0)).save(output, format="PNG")
    return output.getvalue()


def test_uploaded_image_uses_expected_grayscale_preprocessing():
    batch = preprocess_uploaded_image(_png_bytes(), DummyBinaryModel())

    assert tuple(batch.shape) == (1, 8, 8, 1)
    assert float(batch.numpy().min()) == pytest.approx(128 / 255, abs=0.01)
    assert float(batch.numpy().max()) == pytest.approx(128 / 255, abs=0.01)


def test_prediction_uses_selected_threshold_and_returns_probabilities():
    result = predict_uploaded_image(_png_bytes(), DummyBinaryModel(), threshold=0.8)

    assert result["prediction"] == "NORMAL"
    assert result["probability_pneumonia"] == pytest.approx(0.75)
    assert result["probability_normal"] == pytest.approx(0.25)
    assert result["threshold"] == pytest.approx(0.8)


def test_prediction_rejects_non_image_upload():
    with pytest.raises(ValueError, match="readable"):
        preprocess_uploaded_image(b"not an image", DummyBinaryModel())


def test_uploaded_image_uses_mobilenet_rgb_preprocessing():
    batch = preprocess_uploaded_image(_color_png_bytes(), DummyRgbModel())

    assert tuple(batch.shape) == (1, 8, 8, 3)
    assert np.allclose(batch.numpy()[..., 0], batch.numpy()[..., 1])
    assert np.allclose(batch.numpy()[..., 1], batch.numpy()[..., 2])


def test_selected_model_settings_resolve_relative_checkpoint(tmp_path):
    outputs = tmp_path / "outputs"
    models = tmp_path / "models"
    outputs.mkdir()
    models.mkdir()
    checkpoint = models / "selected.keras"
    checkpoint.touch()
    settings_path = outputs / "selected_threshold.json"
    settings_path.write_text(
        json.dumps({"checkpoint": "models/selected.keras", "threshold": 0.72}),
        encoding="utf-8",
    )

    selected_checkpoint, threshold = load_prediction_settings(settings_path)

    assert selected_checkpoint == checkpoint
    assert threshold == pytest.approx(0.72)
