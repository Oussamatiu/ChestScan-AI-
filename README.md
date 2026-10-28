# ChestScan AI

A TensorFlow learning project for classifying chest X-ray images as `NORMAL` or
`PNEUMONIA`. It includes the original custom-CNN command-line pipeline and a
cleaned MobileNetV2 notebook with separate transfer-learning and fine-tuning
experiments. This is an educational prototype, not a medical diagnostic tool.

## Project layout

```text
app.py
streamlit_app.py
configs/config.py
src/
  data_loader.py
  preprocessing.py
  model.py
  train.py
  evaluate.py
  predict.py
models/       # trained model is created here
outputs/      # evaluation artifacts are created here
tests/
mlruns/       # reserved for optional future experiment tracking
```

## Install

Use Python 3.10 or later in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dataset

The dataset must contain `train/` and `test/` folders, each with `NORMAL/` and
`PNEUMONIA/` subfolders. The Kaggle Chest X-Ray Pneumonia dataset can be
downloaded automatically:

```powershell
python app.py train --download
```

To use a dataset already on disk, pass its root directory (the directory that
contains `train/` and `test/`):

```powershell
python app.py train --dataset-dir C:\path\to\chest_xray
```

Training reserves a stratified portion of the training split for validation.
The test split is held out for final evaluation. Training saves the best model
to `models/best_model.keras` by default and then evaluates it.

### Clean MobileNetV2 notebook

Open `notebook/notebook.ipynb` for the simplified MobileNetV2
experiment. It keeps the official test split out of training and threshold
selection in the new workflow, saves distinct checkpoints to
`models/mobilenetv2_transfer_best.keras` and
`models/mobilenetv2_finetune_best.keras`, and writes a selected-threshold
manifest after evaluation. The original notebook is preserved byte-for-byte at
`notebook/original_notebook_backup.ipynb`.

The notebook downloads the dataset with KaggleHub when its dataset setup cell
is run. This does not happen during notebook validation.

The workspace did not contain an extracted dataset or trained model when this
was prepared. Training and metrics are generated only when you execute the
notebook; no historical results have been invented.

## Evaluate and predict

```powershell
python app.py evaluate --dataset-dir C:\path\to\chest_xray
python app.py predict C:\path\to\image.jpg
```

Evaluation writes `outputs/metrics.json` and
`outputs/confusion_matrix.png`. A model file is created only after training;
it is intentionally not checked in as a placeholder.

### Streamlit inference UI

After training and running the notebook's validation-threshold selection and
final-evaluation cells, start the upload interface with:

```powershell
python -m streamlit run streamlit_app.py
```

The UI uses the notebook-selected checkpoint and validation threshold. It
validates image uploads, caches the loaded model, and applies preprocessing
based on the model input shape. It is not clinically validated and does not
replace professional medical advice.

MLflow tracking is included as an optional notebook section. When MLflow is
installed, it logs validation results for both phases and final test metrics for
the selected checkpoint. MLflow runs are created only after you execute the
relevant notebook cells.

## Tests

```powershell
python -m pytest
```
