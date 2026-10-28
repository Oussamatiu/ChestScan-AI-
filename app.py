import argparse
import json
from pathlib import Path

import tensorflow as tf

from configs.config import (
    DATA_DIR,
    EPOCHS,
    IMAGE_SIZE,
    METRICS_PATH,
    MODEL_PATH,
)
from src.data_loader import download_dataset, load_datasets
from src.evaluate import evaluate_model
from src.predict import predict_image
from src.train import train_model


def _dataset_path(args):
    return download_dataset() if args.download else args.dataset_dir


def main():
    parser = argparse.ArgumentParser(description="ChestScan AI image classifier")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train and evaluate the CNN")
    train_parser.add_argument("--dataset-dir", type=Path, default=DATA_DIR)
    train_parser.add_argument("--download", action="store_true")
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--model", type=Path, default=MODEL_PATH)

    evaluate_parser = commands.add_parser("evaluate", help="Evaluate a saved model")
    evaluate_parser.add_argument("--dataset-dir", type=Path, default=DATA_DIR)
    evaluate_parser.add_argument("--download", action="store_true")
    evaluate_parser.add_argument("--model", type=Path, default=MODEL_PATH)

    predict_parser = commands.add_parser("predict", help="Classify one chest X-ray")
    predict_parser.add_argument("image", type=Path)
    predict_parser.add_argument("--model", type=Path, default=MODEL_PATH)

    args = parser.parse_args()

    if args.command == "predict":
        result = predict_image(args.image, args.model)
        print(json.dumps(result, indent=2))
        return

    if args.command == "train" and args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be a positive integer")

    datasets = load_datasets(_dataset_path(args), IMAGE_SIZE)
    if args.command == "train":
        train_model(
            datasets,
            model_path=args.model,
            image_size=IMAGE_SIZE,
            epochs=args.epochs or EPOCHS,
        )

    model_path = args.model
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    metrics, _, _ = evaluate_model(
        model,
        datasets.test,
        datasets.test_labels,
        metrics_path=METRICS_PATH,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
