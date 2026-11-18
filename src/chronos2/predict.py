#predict using loadded models (load_model.py & load_tunned_model.py)

import numpy as np

CONTEXT_LENGTH = 24
PREDICTION_LENGTH = 24

QUANTILE_LEVELS = [0.1, 0.5, 0.9]

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