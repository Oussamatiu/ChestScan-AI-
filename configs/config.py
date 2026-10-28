from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_PATH = MODEL_DIR / "best_model.keras"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"

CLASS_NAMES = ("NORMAL", "PNEUMONIA")
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.15
SEED = 42
THRESHOLD = 0.5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
