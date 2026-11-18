"""Fine-tune Chronos-2 using the existing train/validation pairs."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from packaging.version import Version

from chronos2.load_model import (
    CONTEXT_LENGTH,
    PREDICTION_LENGTH,
    MODEL_NAME,
    load_model,
)
from chronos2.load_tunned_model import FINETUNED_MODEL_PATH

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT_DIR / "data" / "processed" / "forecasting_dataset.npz"
MODELS_DIR = ROOT_DIR / "results" / "models"
RUNS_DIR = MODELS_DIR / "training_runs"

# Conservative starting settings for a desktop GPU. These can be overridden
# from the command line in src.main.
DEFAULT_FINETUNE_MODE = "full"
DEFAULT_NUM_STEPS = 300
DEFAULT_BATCH_SIZE = 8
DEFAULT_LORA_LEARNING_RATE = 1e-5
DEFAULT_FULL_LEARNING_RATE = 1e-6


def _check_finetuning_dependencies(finetune_mode: str) -> None:
    try:
        installed_version = Version(version("chronos-forecasting"))
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "chronos-forecasting is not installed in the active environment."
        ) from exc

    if installed_version < Version("2.2.0"):
        raise RuntimeError(
            "LoRA fine-tuning requires chronos-forecasting >= 2.2.0. "
            "Upgrade with:\n"
            'python -m pip install --upgrade "chronos-forecasting>=2.2.0"'
        )

    if finetune_mode == "lora":
        try:
            import peft  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "LoRA fine-tuning requires peft. Install it with:\n"
                'python -m pip install "peft>=0.18.1,<1"'
            ) from exc


def _load_training_pairs():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATASET_PATH}\n"
            "Run the data preparation script first."
        )

    with np.load(DATASET_PATH) as data:
        required_keys = {"X_train", "y_train", "X_val", "y_val"}
        missing = required_keys.difference(data.files)
        if missing:
            raise KeyError(
                f"Dataset is missing arrays: {sorted(missing)}. "
                f"Available arrays: {data.files}"
            )

        X_train = data["X_train"].astype(np.float32)
        y_train = data["y_train"].astype(np.float32)
        X_val = data["X_val"].astype(np.float32)
        y_val = data["y_val"].astype(np.float32)

    expected_shape_tail = (CONTEXT_LENGTH,)
    for name, array in (
        ("X_train", X_train),
        ("y_train", y_train),
        ("X_val", X_val),
        ("y_val", y_val),
    ):
        if array.ndim != 2 or array.shape[1:] != expected_shape_tail:
            raise ValueError(
                f"{name} must have shape (samples, {CONTEXT_LENGTH}), "
                f"received {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains missing or non-finite values.")

    if len(X_train) != len(y_train) or len(X_val) != len(y_val):
        raise ValueError("Input and target arrays have different sample counts.")
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Training and validation sets must both be non-empty.")

    # Chronos2Pipeline.fit accepts complete time series. Each concatenated
    # 48-hour sequence contains exactly one context/target pair:
    # first 24 values = context, next 24 values = forecast target.
    train_inputs = [
        np.concatenate((x, y)).astype(np.float32, copy=False)
        for x, y in zip(X_train, y_train)
    ]
    validation_inputs = [
        np.concatenate((x, y)).astype(np.float32, copy=False)
        for x, y in zip(X_val, y_val)
    ]

    expected_length = CONTEXT_LENGTH + PREDICTION_LENGTH
    if any(len(series) != expected_length for series in train_inputs + validation_inputs):
        raise ValueError("Every fine-tuning sequence must contain exactly 48 values.")

    return train_inputs, validation_inputs


def _save_final_model(pipeline, metadata: dict) -> None:
    """Save to a temporary directory, then replace the active checkpoint."""
    temp_path = FINETUNED_MODEL_PATH.with_name(FINETUNED_MODEL_PATH.name + "_new")
    backup_path = FINETUNED_MODEL_PATH.with_name(FINETUNED_MODEL_PATH.name + "_backup")

    if temp_path.exists():
        shutil.rmtree(temp_path)
    if backup_path.exists():
        shutil.rmtree(backup_path)

    pipeline.save_pretrained(temp_path)
    with (temp_path / "finetune_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    moved_old_model = False
    try:
        if FINETUNED_MODEL_PATH.exists():
            FINETUNED_MODEL_PATH.replace(backup_path)
            moved_old_model = True
        temp_path.replace(FINETUNED_MODEL_PATH)
    except Exception:
        if moved_old_model and backup_path.exists() and not FINETUNED_MODEL_PATH.exists():
            backup_path.replace(FINETUNED_MODEL_PATH)
        raise
    else:
        if backup_path.exists():
            shutil.rmtree(backup_path)


def finetune(
    finetune_mode: str = DEFAULT_FINETUNE_MODE,
    num_steps: int = DEFAULT_NUM_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float | None = None,
):
    if finetune_mode not in {"lora", "full"}:
        raise ValueError("finetune_mode must be 'lora' or 'full'.")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    _check_finetuning_dependencies(finetune_mode)

    if learning_rate is None:
        learning_rate = (
            DEFAULT_LORA_LEARNING_RATE
            if finetune_mode == "lora"
            else DEFAULT_FULL_LEARNING_RATE
        )

    train_inputs, validation_inputs = _load_training_pairs()

    print("=== CHRONOS-2 FINE-TUNING ===")
    print(f"Base model: {MODEL_NAME}")
    print(f"Fine-tuning mode: {finetune_mode}")
    print(f"Training samples: {len(train_inputs)}")
    print(f"Validation samples: {len(validation_inputs)}")
    print(f"Context length: {CONTEXT_LENGTH}")
    print(f"Prediction length: {PREDICTION_LENGTH}")
    print(f"Steps: {num_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate:g}")

    base_pipeline = load_model()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"chronos2_{finetune_mode}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # fit() fine-tunes a copy of the loaded model and returns a new pipeline.
    # Validation is used by the Trainer for checkpoint/model selection.
    finetuned_pipeline = base_pipeline.fit(
        inputs=train_inputs,
        validation_inputs=validation_inputs,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        min_past=CONTEXT_LENGTH,
        finetune_mode=finetune_mode,
        learning_rate=learning_rate,
        num_steps=num_steps,
        batch_size=batch_size,
        output_dir=run_dir,
        finetuned_ckpt_name="finetuned-ckpt",
        seed=42,
        data_seed=42,
    )

    metadata = {
        "base_model": MODEL_NAME,
        "fine_tuning_mode": finetune_mode,
        "context_length": CONTEXT_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "random_seed": 42,
        "train_samples": len(train_inputs),
        "validation_samples": len(validation_inputs),
        "source_dataset": str(DATASET_PATH.relative_to(ROOT_DIR)),
        "training_run_directory": str(run_dir.relative_to(ROOT_DIR)),
    }

    _save_final_model(finetuned_pipeline, metadata)

    print("\nFine-tuning completed.")
    print("Final model saved to:", FINETUNED_MODEL_PATH)
    print("Training logs/checkpoints saved to:", run_dir)

    return FINETUNED_MODEL_PATH


if __name__ == "__main__":
    finetune()
