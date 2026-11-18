#forecast for tunned models

from pathlib import Path

import torch
from chronos import Chronos2Pipeline

ROOT_DIR = Path(__file__).resolve().parents[2]

FINETUNED_MODEL_PATHS = {
    "full": ROOT_DIR / "results" / "models" / "chronos2_finetuned_full",
    "lora": ROOT_DIR / "results" / "models" / "chronos2_finetuned_lora",
}

def load_model(mode: str = "full"):
    mode = mode.lower()

    if mode not in FINETUNED_MODEL_PATHS:
        raise ValueError(
            f"Unknown fine-tuning mode: {mode}. "
            f"Available modes: {', '.join(FINETUNED_MODEL_PATHS)}"
        )

    model_path = FINETUNED_MODEL_PATHS[mode]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned Chronos-2 ({mode}) model was not found.\n"
            f"Run first:\n"
            f"python -m src.main finetune --mode {mode}\n"
            f"Expected checkpoint directory: {model_path}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading fine-tuned Chronos-2 ({mode}) from: {model_path}")
    print(f"Device: {device}")

    pipeline = Chronos2Pipeline.from_pretrained(
        str(model_path),
        device_map=device,
    )

    print(f"Fine-tuned Chronos-2 ({mode}) loaded successfully.")

    return pipeline