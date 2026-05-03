# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

PyTorch transfer-learning pipeline that classifies brain MRI scans into 4 classes (`glioma`, `meningioma`, `notumor`, `pituitary`). The shipped model is EfficientNet-B0 (~94.8% test accuracy). The pipeline is config-driven (YAML), tracked with MLflow, and served to the .NET backend over a FastAPI HTTP layer.

## Setup

PyTorch must be installed first from its own index — `requirements.txt` does not pin it, because the right wheel depends on the user's CUDA setup.

```bash
# 1. PyTorch (pick GPU or CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # NVIDIA CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu     # CPU only

# 2. Everything else
pip install -r requirements.txt
pip install -e .   # installs the `src` package so `from src.x import y` works in scripts
```

Python **3.11 or 3.12** only — PyTorch does not yet ship wheels for 3.13+.

## Common commands

All scripts assume the project root is the working directory and rely on the editable install (`pip install -e .`) for `src.*` imports.

```bash
# One-time: clean + split the raw Kaggle dataset into data/splits/{train,val,test}/<class>/
python scripts/prepare_data.py

# Train (default config = ResNet-18 baseline). Pass --config for other backbones.
python scripts/train.py
python scripts/train.py --config configs/efficientnet.yaml
python scripts/train.py --config configs/densenet.yaml

# Evaluate on the test split. Reads backbone + data paths from the checkpoint's saved config.
python scripts/evaluate.py
python scripts/evaluate.py --model outputs/models/best_model.pth

# Bundle a checkpoint + metadata + standalone usage_example.py into outputs/deployment/
python scripts/export_model.py --model outputs/models/best_model.pth

# Serve the model over HTTP (FastAPI + uvicorn). Swagger UI at /docs.
python scripts/run_api.py                 # 0.0.0.0:8000
python scripts/run_api.py --reload        # dev: auto-reload on code change

# View MLflow runs (training writes to a SQLite store under outputs/logs/)
mlflow ui --backend-store-uri sqlite:///outputs/logs/mlflow.db

# Latency benchmarks
python scripts/benchmark_predict.py       # in-process predictor
python scripts/benchmark_api.py           # over HTTP

# Smoke tests (these are scripts, not pytest — run individually)
python scripts/test_data_pipeline.py
python scripts/test_model.py
python scripts/test_predictor.py
```

There is no test runner configured. `tests/` is an empty package; the `scripts/test_*.py` files are runnable smoke tests, not pytest cases.

## Architecture

### Layered, config-driven pipeline

```
configs/*.yaml ──► scripts/train.py ──► src.models.classifier   (build model)
                                   ──► src.data.dataset         (build loaders)
                                   ──► src.training.trainer     (fit + checkpoint)
                                                  │
                                                  ▼
                            outputs/models/best_model.pth
                                                  │
                ┌─────────────────────────────────┼─────────────────────────────────┐
                ▼                                 ▼                                 ▼
   scripts/evaluate.py                 scripts/export_model.py             src.api.main:app
   (test-set metrics + plots)          (outputs/deployment/ bundle)        (FastAPI HTTP server)
                                                                                   │
                                                                                   ▼
                                                                    src.inference.predictor
                                                                    (the single shared inference path)
```

The same `BrainTumorPredictor` is used by the FastAPI app, the standalone `usage_example.py` in the deployment bundle, and the benchmark scripts — keep them consistent if you change inference behavior.

### Cross-cutting invariants — read these before changing things

- **Class order is load-bearing.** `CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]` is duplicated in `src/data/dataset.py`, `src/inference/predictor.py`, and `scripts/export_model.py`. `dataset.py` *asserts* that `ImageFolder.classes` matches this exact order — alphabetical sort of the directory names happens to coincide. If you rename a class folder, update all three locations and re-verify the assertion.
- **Checkpoint schema.** `Trainer.save_checkpoint` writes a dict with keys `model_state_dict`, `backbone_name`, `num_classes`, `val_accuracy`, `config`. Both `evaluate.py` and `BrainTumorPredictor` rebuild the model from `backbone_name` + `num_classes` and rely on the embedded `config` for data paths. Don't drop fields.
- **Backbone registry.** `BrainTumorClassifier.BACKBONES` is the single source of truth for supported models and their feature dimensions. Adding a backbone means: add the entry, plus a branch in `__init__` for how to strip the original head, plus (if it needs pooling) a branch in `forward`.
- **Multiple backbone branches in `forward`.** ResNet variants reuse the original `avgpool` (it's part of `children()[:-1]`), but DenseNet/EfficientNet need an explicit `self.pool`. New backbones must follow whichever pattern matches their structure.
- **Mixed precision is CUDA-only.** `Trainer` constructs `GradScaler("cuda")` and uses `autocast("cuda")` unconditionally — training on CPU will not work without changes. Inference (`predictor.py`) is device-agnostic.
- **Default config trap.** `scripts/train.py` defaults to `configs/default.yaml` (ResNet-18). The shipped/best model uses `configs/efficientnet.yaml` — pass `--config` explicitly when reproducing the 94.8% number.
- **Augmentation toggle.** `data.strong_augmentation: true` in YAML switches `get_train_transforms` → `get_strong_train_transforms`. EfficientNet config uses strong; default does not.
- **Trainer always overwrites `outputs/models/best_model.pth`.** It does not include the backbone name in the filename. If you train multiple backbones back-to-back, rename or move checkpoints between runs or you will lose them.
- **API model path.** `src/api/main.py` reads `MODEL_PATH` env var (default `outputs/models/best_model.pth`). The model is loaded once in the `lifespan` startup hook and reused for every request. CORS is `allow_origins=["*"]` — tighten for production.

### Layout

- `src/data/` — `ImageFolder`-based dataset + train/val/strong-aug transforms. `data_dir` points at the splits directory created by `prepare_data.py`.
- `src/models/classifier.py` — `BrainTumorClassifier` with multi-backbone registry and a shared `Flatten → Dropout → Linear → ReLU → Dropout → Linear` head.
- `src/training/trainer.py` — fit loop, AMP, StepLR, best-checkpoint tracking. Owns the checkpoint format.
- `src/evaluation/metrics.py` — `get_predictions`, classification report, confusion matrix and misclassification plots used by `evaluate.py`.
- `src/inference/predictor.py` — the public inference interface. Accepts a path **or** a `PIL.Image`.
- `src/api/main.py` — FastAPI app with `GET /health`, `POST /predict` (multipart upload, 10 MB cap, JPEG/PNG/BMP/TIFF only). See `docs/API.md` for the request/response contract and the C# client snippet handed to the .NET team.
- `configs/{default,efficientnet,densenet}.yaml` — one file per experiment. Same schema across all three.
- `data/` and `outputs/` are gitignored; both are required at runtime and created on demand by `prepare_data.py` / `train.py`.

## Conventions

- Project root is always the working directory when running scripts; imports are `from src.x.y import Z`.
- New experiments = new YAML in `configs/`, not new code paths. The only switches the trainer reads are listed in `configs/efficientnet.yaml`.
- Keep the inference path single-sourced through `BrainTumorPredictor`. Don't reimplement preprocessing in new entry points — call `get_val_transforms()` or use the predictor.
