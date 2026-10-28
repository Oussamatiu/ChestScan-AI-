import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from configs.config import EPOCHS, MODEL_PATH, SEED
from src.model import build_model


def train_model(
    datasets,
    model_path=MODEL_PATH,
    image_size=(180, 180),
    epochs=EPOCHS,
):
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(image_size)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    history = model.fit(
        datasets.train,
        validation_data=datasets.validation,
        epochs=epochs,
        class_weight=datasets.class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    return history
