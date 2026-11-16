from pathlib import Path

import torch
from chronos import Chronos2Pipeline

from src.chronos2.forecast_model import (
    CONTEXT_LENGTH,
    PREDICTION_LENGTH,
    QUANTILE_LEVELS,
    predict_next_day,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
FINETUNED_MODEL_PATH = ROOT_DIR / "results" / "models" / "chronos2_finetuned"


def load_model():
    """Load the locally fine-tuned Chronos-2 checkpoint."""
    if not FINETUNED_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Fine-tuned Chronos-2 model was not found. Run first:\n"
            "python -m src.main finetune\n"
            f"Expected checkpoint directory: {FINETUNED_MODEL_PATH}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading fine-tuned Chronos-2 from: {FINETUNED_MODEL_PATH}")
    print(f"Device: {device}")

    pipeline = Chronos2Pipeline.from_pretrained(
        str(FINETUNED_MODEL_PATH),
        device_map=device,
    )

    print("Fine-tuned model loaded successfully.")
    return pipeline
