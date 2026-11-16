from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.chronos2.forecast_model import (
    PREDICTION_LENGTH,
    QUANTILE_LEVELS,
    load_model as load_zero_shot_model,
)
from src.chronos2.finetuned_forecast_model import (
    load_model as load_finetuned_model,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "data" / "processed" / "forecasting_dataset.npz"
RESULTS_DIR = ROOT_DIR / "results"
FORECASTS_DIR = RESULTS_DIR / "forecasts"
PLOTS_DIR = RESULTS_DIR / "plots"
BATCH_SIZE = 16

MODEL_LOADERS = {
    "zero-shot": ("Chronos-2 zero-shot", load_zero_shot_model),
    "finetuned": ("Chronos-2 fine-tuned", load_finetuned_model),
}


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return float(mae), float(rmse), float(r2)


def predict_dataset(pipeline, X):
    predictions = np.zeros((len(X), PREDICTION_LENGTH), dtype=np.float32)
    quantile_predictions = np.zeros(
        (len(X), PREDICTION_LENGTH, len(QUANTILE_LEVELS)),
        dtype=np.float32,
    )

    for start in range(0, len(X), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(X))
        batch_inputs = [sample for sample in X[start:end]]

        quantiles, point_forecasts = pipeline.predict_quantiles(
            inputs=batch_inputs,
            prediction_length=PREDICTION_LENGTH,
            quantile_levels=QUANTILE_LEVELS,
            batch_size=BATCH_SIZE,
        )

        for i in range(len(batch_inputs)):
            predictions[start + i] = point_forecasts[i][0].detach().cpu().numpy()
            quantile_predictions[start + i] = quantiles[i][0].detach().cpu().numpy()

        print(f"Processed {end}/{len(X)} samples")

    return predictions, quantile_predictions


def _load_test_data():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {DATASET_PATH}")

    with np.load(DATASET_PATH) as data:
        X_test = data["X_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.float32)

    if len(X_test) == 0:
        raise ValueError("Test dataset is empty.")
    if X_test.ndim != 2 or X_test.shape[1] != PREDICTION_LENGTH:
        raise ValueError(f"Unexpected X_test shape: {X_test.shape}")
    if y_test.shape != X_test.shape:
        raise ValueError(f"X_test/y_test shapes differ: {X_test.shape} vs {y_test.shape}")
    if not np.isfinite(X_test).all() or not np.isfinite(y_test).all():
        raise ValueError("Test data contains missing or non-finite values.")

    return X_test, y_test


def evaluate(model_type="zero-shot"):
    """Evaluate zero-shot, fine-tuned, or both models on the same test split."""
    if model_type not in {"zero-shot", "finetuned", "both"}:
        raise ValueError("model_type must be 'zero-shot', 'finetuned', or 'both'.")

    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X_test, y_test = _load_test_data()

    print("=== TEST DATA ===")
    print(f"Samples: {len(X_test)}")
    print(f"Input shape: {X_test.shape}")
    print(f"Target shape: {y_test.shape}")

    selected_models = ["zero-shot", "finetuned"] if model_type == "both" else [model_type]
    model_predictions = {}
    metrics_rows = []

    for model_key in selected_models:
        model_label, loader = MODEL_LOADERS[model_key]
        print(f"\n=== {model_label.upper()} ===")

        pipeline = loader()
        predictions, quantiles = predict_dataset(pipeline, X_test)
        model_predictions[model_key] = {
            "predictions": predictions,
            "quantiles": quantiles,
            "label": model_label,
        }

        metrics = calculate_metrics(y_test, predictions)
        metrics_rows.append({
            "model": model_label,
            "MAE_kW": metrics[0],
            "RMSE_kW": metrics[1],
            "R2": metrics[2],
        })

        print(f"MAE:  {metrics[0]:.4f} kW")
        print(f"RMSE: {metrics[1]:.4f} kW")
        print(f"R2:   {metrics[2]:.4f}")

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Seasonal Naive: repeat the previous day's profile.
    baseline_predictions = X_test.copy()
    baseline_metrics = calculate_metrics(y_test, baseline_predictions)
    metrics_rows.append({
        "model": "Seasonal Naive",
        "MAE_kW": baseline_metrics[0],
        "RMSE_kW": baseline_metrics[1],
        "R2": baseline_metrics[2],
    })

    print("\n=== SEASONAL NAIVE ===")
    print(f"MAE:  {baseline_metrics[0]:.4f} kW")
    print(f"RMSE: {baseline_metrics[1]:.4f} kW")
    print(f"R2:   {baseline_metrics[2]:.4f}")

    # Keep the original zero-shot filenames and columns for reproducibility.
    if model_type == "zero-shot":
        output_path = FORECASTS_DIR / "test_predictions.csv"
        plot_path = PLOTS_DIR / "test_average_profile.png"
    elif model_type == "finetuned":
        output_path = FORECASTS_DIR / "test_predictions_finetuned.csv"
        plot_path = PLOTS_DIR / "test_average_profile_finetuned.png"
    else:
        output_path = FORECASTS_DIR / "test_predictions_comparison.csv"
        plot_path = PLOTS_DIR / "test_average_profile_comparison.png"

    output_data = {
        "sample": np.repeat(np.arange(len(y_test)), PREDICTION_LENGTH),
        "hour_ahead": np.tile(np.arange(1, PREDICTION_LENGTH + 1), len(y_test)),
        "actual_kw": y_test.reshape(-1),
        "seasonal_naive_kw": baseline_predictions.reshape(-1),
    }

    if len(selected_models) == 1:
        model_data = model_predictions[selected_models[0]]
        output_data["prediction_kw"] = model_data["predictions"].reshape(-1)
        output_data["q10_kw"] = model_data["quantiles"][:, :, 0].reshape(-1)
        output_data["q50_kw"] = model_data["quantiles"][:, :, 1].reshape(-1)
        output_data["q90_kw"] = model_data["quantiles"][:, :, 2].reshape(-1)
    else:
        for model_key, short_name in (("zero-shot", "zero_shot"), ("finetuned", "finetuned")):
            model_data = model_predictions[model_key]
            output_data[f"{short_name}_prediction_kw"] = model_data["predictions"].reshape(-1)
            output_data[f"{short_name}_q10_kw"] = model_data["quantiles"][:, :, 0].reshape(-1)
            output_data[f"{short_name}_q50_kw"] = model_data["quantiles"][:, :, 1].reshape(-1)
            output_data[f"{short_name}_q90_kw"] = model_data["quantiles"][:, :, 2].reshape(-1)

    pd.DataFrame(output_data).to_csv(output_path, index=False)
    metrics_path = FORECASTS_DIR / f"test_metrics_{model_type.replace('-', '_')}.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    # Compare mean profiles over all test examples.
    hours = np.arange(PREDICTION_LENGTH)
    plt.figure(figsize=(11, 5))
    plt.plot(hours, y_test.mean(axis=0), label="Actual", linewidth=2)
    plt.plot(hours, baseline_predictions.mean(axis=0), label="Seasonal Naive", linewidth=2)

    for model_key in selected_models:
        predictions = model_predictions[model_key]["predictions"]
        plt.plot(hours, predictions.mean(axis=0), label=model_predictions[model_key]["label"])

    plt.xlabel("Hour of day")
    plt.ylabel("Active power (kW)")
    plt.title("Average daily power profile")
    plt.xticks(hours)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\nSaved predictions:", output_path)
    print("Saved metrics:", metrics_path)
    print("Saved plot:", plot_path)


if __name__ == "__main__":
    evaluate()
