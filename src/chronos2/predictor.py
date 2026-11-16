from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.chronos2.forecast_model import (
    load_model,
    predict_next_day,
    CONTEXT_LENGTH,
    PREDICTION_LENGTH
)


ROOT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT_DIR / "data" / "processed" / "hourly_power_raw.csv"
)

FORECASTS_DIR = ROOT_DIR / "results" / "forecasts"
PLOTS_DIR = ROOT_DIR / "results" / "plots"

MIN_VALID_MINUTES = 15


def load_latest_context():
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["datetime"],
        index_col="datetime"
    )

    df = df.sort_index().asfreq("1h")

    power = pd.to_numeric(df["power"], errors="coerce")

    valid_minutes = pd.to_numeric(
        df["valid_minutes"], errors="coerce"
    ).fillna(0)

    # Reject hours with insufficient measurements.
    power = power.where(valid_minutes >= MIN_VALID_MINUTES)

    values = power.to_numpy(dtype=np.float32)
    valid = np.isfinite(values)

    # Find continuous valid segments.
    edges = np.diff(
        np.concatenate(([False], valid, [False])).astype(np.int8)
    )

    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    # Find the latest segment that contains at least 24 hours.
    for start, end in reversed(list(zip(starts, ends))):
        if end - start >= CONTEXT_LENGTH:
            context = power.iloc[end - CONTEXT_LENGTH:end]

            return context

    raise ValueError(
        "No continuous segment with 24 valid hours was found."
    )


def run_forecast():
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Get the latest 24 valid hours.
    context = load_latest_context()

    print("=== INPUT DATA ===")
    print(f"From: {context.index[0]}")
    print(f"To:   {context.index[-1]}")

    # Load Chronos-2.
    pipeline = load_model()

    # Generate the next 24 hours.
    prediction, quantiles = predict_next_day(
        pipeline,
        context.to_numpy()
    )

    future_index = pd.date_range(
        start=context.index[-1] + pd.Timedelta(hours=1),
        periods=PREDICTION_LENGTH,
        freq="1h"
    )

    forecast_df = pd.DataFrame({
        "datetime": future_index,
        "prediction_kw": prediction,
        "q10_kw": quantiles[:, 0],
        "q50_kw": quantiles[:, 1],
        "q90_kw": quantiles[:, 2]
    })

    # Save forecast.
    output_path = FORECASTS_DIR / "latest_forecast.csv"

    forecast_df.to_csv(output_path, index=False)

    print("\n=== ELECTRICITY FORECAST ===")

    for _, row in forecast_df.iterrows():
        print(
            f"{row['datetime']:%Y-%m-%d %H:%M}  "
            f"{row['prediction_kw']:.3f} kW"
        )

    # Plot historical context and forecast.
    plt.figure(figsize=(12, 5))

    plt.plot(
        context.index,
        context.values,
        label="Historical consumption"
    )

    plt.plot(
        future_index,
        prediction,
        label="Forecast",
        linestyle="--"
    )

    plt.fill_between(
        future_index,
        quantiles[:, 0],
        quantiles[:, 2],
        alpha=0.2,
        label="10-90% quantile range"
    )

    plt.xlabel("Datetime")
    plt.ylabel("Active power (kW)")
    plt.title("Electricity consumption forecast")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = PLOTS_DIR / "latest_forecast.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\nSaved:", output_path)
    print("Saved:", plot_path)


if __name__ == "__main__":
    run_forecast()
