#forecast for zeroshot model

import torch

from chronos import Chronos2Pipeline

MODEL_NAME = "amazon/chronos-2"

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {device}")

    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_NAME,
        device_map=device,
    )

    print("Model loaded successfully.")
    return pipeline
