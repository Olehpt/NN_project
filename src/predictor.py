from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.chronos2.forecast_model import (
    CONTEXT_LENGTH,
    PREDICTION_LENGTH,
    predict_next_day,
    load_model as load_zero_shot_model,
)
from src.chronos2.finetuned_forecast_model import (
    load_model as load_finetuned_model,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "data" / "processed" / "hourly_power_raw.csv"
FORECASTS_DIR = ROOT_DIR / "results" / "forecasts"
PLOTS_DIR = ROOT_DIR / "results" / "plots"
MIN_VALID_MINUTES = 15

MODEL_LOADERS = {
    "zero-shot": ("Chronos-2 zero-shot", load_zero_shot_model),
    "finetuned": ("Chronos-2 fine-tuned", load_finetuned_model),
}


def load_latest_context():
    df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"], index_col="datetime")
    df = df.sort_index().asfreq("1h")

    power = pd.to_numeric(df["power"], errors="coerce")
    valid_minutes = pd.to_numeric(df["valid_minutes"], errors="coerce").fillna(0)
    power = power.where(valid_minutes >= MIN_VALID_MINUTES)

    values = power.to_numpy(dtype=np.float32)
    valid = np.isfinite(values)
    edges = np.diff(np.concatenate(([False], valid, [False])).astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    for start, end in reversed(list(zip(starts, ends))):
        if end - start >= CONTEXT_LENGTH:
            return power.iloc[end - CONTEXT_LENGTH:end]

    raise ValueError(
        f"No continuous segment with {CONTEXT_LENGTH} valid hours was found."
    )


def run_forecast(model_type="zero-shot"):
    if model_type not in MODEL_LOADERS:
        raise ValueError("model_type must be 'zero-shot' or 'finetuned'.")

    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    context = load_latest_context()
    model_label, loader = MODEL_LOADERS[model_type]

    print("=== INPUT DATA ===")
    print(f"Model: {model_label}")
    print(f"From:  {context.index[0]}")
    print(f"To:    {context.index[-1]}")

    pipeline = loader()
    prediction, quantiles = predict_next_day(pipeline, context.to_numpy())

    future_index = pd.date_range(
        start=context.index[-1] + pd.Timedelta(hours=1),
        periods=PREDICTION_LENGTH,
        freq="1h",
    )

    forecast_df = pd.DataFrame({
        "datetime": future_index,
        "prediction_kw": prediction,
        "q10_kw": quantiles[:, 0],
        "q50_kw": quantiles[:, 1],
        "q90_kw": quantiles[:, 2],
    })

    suffix = "finetuned" if model_type == "finetuned" else "zero_shot"
    output_path = FORECASTS_DIR / f"latest_forecast_{suffix}.csv"
    forecast_df.to_csv(output_path, index=False)

    print("\n=== ELECTRICITY FORECAST ===")
    for _, row in forecast_df.iterrows():
        print(f"{row['datetime']:%Y-%m-%d %H:%M}  {row['prediction_kw']:.3f} kW")

    plt.figure(figsize=(12, 5))
    plt.plot(context.index, context.values, label="Historical consumption")
    plt.plot(future_index, prediction, label=f"Forecast ({model_label})", linestyle="--")
    plt.fill_between(
        future_index,
        quantiles[:, 0],
        quantiles[:, 2],
        alpha=0.2,
        label="10-90% quantile range",
    )
    plt.xlabel("Datetime")
    plt.ylabel("Active power (kW)")
    plt.title("Electricity consumption forecast")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = PLOTS_DIR / f"latest_forecast_{suffix}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\nSaved:", output_path)
    print("Saved:", plot_path)


if __name__ == "__main__":
    run_forecast()
