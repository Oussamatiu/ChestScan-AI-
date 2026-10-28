from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from configs.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    SEED,
    VALIDATION_SPLIT,
)
from src.preprocessing import preprocess_image


@dataclass
class DatasetBundle:
    train: tf.data.Dataset
    validation: tf.data.Dataset
    test: tf.data.Dataset
    test_paths: list[Path]
    test_labels: np.ndarray
    class_weights: dict[int, float]


def locate_dataset(dataset_dir):
    """Find the directory containing the expected train and test splits."""
    dataset_dir = Path(dataset_dir).expanduser()
    candidates = (
        dataset_dir / "chest_xray" / "chest_xray",
        dataset_dir / "chest_xray",
        dataset_dir,
    )
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find train/ and test/ directories under {dataset_dir}"
    )


def download_dataset():
    """Download the public Kaggle chest X-ray dataset using kagglehub."""
    import kagglehub

    return locate_dataset(
        kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    )


def collect_images(split_dir):
    paths = []
    labels = []
    split_dir = Path(split_dir)

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        class_images = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        paths.extend(class_images)
        labels.extend([label] * len(class_images))

    if not paths:
        raise ValueError(f"No supported images found under {split_dir}")
    return paths, np.asarray(labels, dtype=np.int32)


def _make_dataset(paths, labels, image_size, training=False):
    path_strings = [str(path) for path in paths]
    dataset = tf.data.Dataset.from_tensor_slices((path_strings, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(path_strings),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda path, label: preprocess_image(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def load_datasets(dataset_dir, image_size):
    """Build shuffled training and deterministic validation/test datasets."""
    data_dir = locate_dataset(dataset_dir)
    all_train_paths, all_train_labels = collect_images(data_dir / "train")
    test_paths, test_labels = collect_images(data_dir / "test")

    train_paths, validation_paths, train_labels, validation_labels = (
        train_test_split(
            all_train_paths,
            all_train_labels,
            test_size=VALIDATION_SPLIT,
            random_state=SEED,
            stratify=all_train_labels,
        )
    )

    classes = np.unique(train_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels,
    )
    class_weights = {
        int(label): float(weight)
        for label, weight in zip(classes, weights)
    }

    return DatasetBundle(
        train=_make_dataset(train_paths, train_labels, image_size, training=True),
        validation=_make_dataset(validation_paths, validation_labels, image_size),
        test=_make_dataset(test_paths, test_labels, image_size),
        test_paths=test_paths,
        test_labels=test_labels,
        class_weights=class_weights,
    )
