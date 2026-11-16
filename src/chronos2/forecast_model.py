import numpy as np
import torch

from chronos import Chronos2Pipeline

MODEL_NAME = "amazon/chronos-2"

CONTEXT_LENGTH = 24
PREDICTION_LENGTH = 24

QUANTILE_LEVELS = [0.1, 0.5, 0.9]


def load_model():
    """Load the original pretrained Chronos-2 model (zero-shot)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {device}")

    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_NAME,
        device_map=device,
    )

    print("Model loaded successfully.")
    return pipeline


def predict_next_day(pipeline, context):
    context = np.asarray(context, dtype=np.float32).reshape(-1)

    if len(context) != CONTEXT_LENGTH:
        raise ValueError(
            f"Expected {CONTEXT_LENGTH} input values, received {len(context)}"
        )

    if not np.isfinite(context).all():
        raise ValueError("Input contains missing or non-finite values.")

    quantiles, point_forecast = pipeline.predict_quantiles(
        inputs=[context],
        prediction_length=PREDICTION_LENGTH,
        quantile_levels=QUANTILE_LEVELS,
    )

    prediction = point_forecast[0][0].detach().cpu().numpy()
    quantile_values = quantiles[0][0].detach().cpu().numpy()

    return prediction, quantile_values
