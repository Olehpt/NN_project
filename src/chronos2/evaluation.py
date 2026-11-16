from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.chronos2.forecast_model import (
    load_model,
    PREDICTION_LENGTH,
    QUANTILE_LEVELS
)


ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    ROOT_DIR / "data" / "processed" / "forecasting_dataset.npz"
)

RESULTS_DIR = ROOT_DIR / "results"
FORECASTS_DIR = RESULTS_DIR / "forecasts"
PLOTS_DIR = RESULTS_DIR / "plots"

BATCH_SIZE = 16


def calculate_metrics(actual, predicted):
    actual = actual.reshape(-1)
    predicted = predicted.reshape(-1)

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return mae, rmse, r2


def evaluate():
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the test split.
    with np.load(DATASET_PATH) as data:
        X_test = data["X_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.float32)

    print("=== TEST DATA ===")
    print(f"Samples: {len(X_test)}")
    print(f"Input shape: {X_test.shape}")
    print(f"Target shape: {y_test.shape}")

    if len(X_test) == 0:
        raise ValueError("Test dataset is empty.")

    pipeline = load_model()

    predictions = np.zeros_like(y_test)

    quantile_predictions = np.zeros(
        (*y_test.shape, len(QUANTILE_LEVELS)),
        dtype=np.float32
    )

    # Process test samples in batches.
    for start in range(0, len(X_test), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(X_test))

        batch_inputs = [
            sample for sample in X_test[start:end]
        ]

        quantiles, point_forecasts = pipeline.predict_quantiles(
            inputs=batch_inputs,
            prediction_length=PREDICTION_LENGTH,
            quantile_levels=QUANTILE_LEVELS,
            batch_size=BATCH_SIZE
        )

        for i in range(len(batch_inputs)):
            predictions[start + i] = (
                point_forecasts[i][0].detach().cpu().numpy()
            )

            quantile_predictions[start + i] = (
                quantiles[i][0].detach().cpu().numpy()
            )

        print(f"Processed {end}/{len(X_test)} samples")

    # Seasonal Naive: repeat the previous day's profile.
    baseline_predictions = X_test.copy()

    # Calculate metrics.
    model_metrics = calculate_metrics(y_test, predictions)
    baseline_metrics = calculate_metrics(y_test, baseline_predictions)

    print("\n=== RESULTS ===")

    print("\nChronos-2:")
    print(f"MAE:  {model_metrics[0]:.4f} kW")
    print(f"RMSE: {model_metrics[1]:.4f} kW")
    print(f"R2:   {model_metrics[2]:.4f}")

    print("\nSeasonal Naive:")
    print(f"MAE:  {baseline_metrics[0]:.4f} kW")
    print(f"RMSE: {baseline_metrics[1]:.4f} kW")
    print(f"R2:   {baseline_metrics[2]:.4f}")

    # Save individual predictions.
    output_df = pd.DataFrame({
        "sample": np.repeat(np.arange(len(y_test)), PREDICTION_LENGTH),
        "hour_ahead": np.tile(
            np.arange(1, PREDICTION_LENGTH + 1),
            len(y_test)
        ),
        "actual_kw": y_test.reshape(-1),
        "prediction_kw": predictions.reshape(-1),
        "q10_kw": quantile_predictions[:, :, 0].reshape(-1),
        "q50_kw": quantile_predictions[:, :, 1].reshape(-1),
        "q90_kw": quantile_predictions[:, :, 2].reshape(-1)
    })

    output_path = FORECASTS_DIR / "test_predictions.csv"
    output_df.to_csv(output_path, index=False)

    # Compare average daily profiles.
    hours = np.arange(PREDICTION_LENGTH)

    plt.figure(figsize=(10, 5))

    plt.plot(
        hours,
        y_test.mean(axis=0),
        label="Actual"
    )

    plt.plot(
        hours,
        predictions.mean(axis=0),
        label="Chronos-2"
    )

    plt.plot(
        hours,
        baseline_predictions.mean(axis=0),
        label="Seasonal Naive"
    )

    plt.xlabel("Hour of day")
    plt.ylabel("Active power (kW)")
    plt.title("Average daily power profile")
    plt.xticks(hours)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = PLOTS_DIR / "test_average_profile.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\nSaved:", output_path)
    print("Saved:", plot_path)


if __name__ == "__main__":
    evaluate()
