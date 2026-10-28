import numpy as np
from PIL import Image

from src.preprocessing import load_image_for_prediction, preprocess_image


def test_preprocess_image_resizes_normalizes_and_preserves_label(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.full((12, 16), 255, dtype=np.uint8)).save(image_path)

    image, label = preprocess_image(str(image_path), 1, (8, 10))

    assert tuple(image.shape) == (8, 10, 1)
    assert float(image.numpy().min()) == 1.0
    assert float(image.numpy().max()) == 1.0
    assert float(label.numpy()) == 1.0


def test_prediction_preprocessing_adds_batch_dimension(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((12, 16), dtype=np.uint8)).save(image_path)

    image = load_image_for_prediction(image_path, (8, 10))

    assert tuple(image.shape) == (1, 8, 10, 1)
    assert float(image.numpy().min()) == 0.0
    assert float(image.numpy().max()) == 0.0
