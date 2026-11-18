"""Single entry point for electricity forecasting, fine-tuning and evaluation.

Run from the project root:
    python -m src.main
    python -m src.main forecast --model zero-shot
    python -m src.main train --model all
    python -m src.main evaluate --model all

With no subcommand, the application forecasts from data/input.txt and writes
its result to data/output.txt.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "input.txt"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "output.txt"

# The existing modules use both `src.*` and top-level `chronos2.*` imports.
# Add both roots so that main.py works as a module and as a directly-run file.
for _path in (str(SRC_DIR), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

MODEL_KEYS = ("zero-shot", "full", "lora")
MODEL_LABELS = {
    "zero-shot": "Chronos-2 zero-shot",
    "full": "Chronos-2 fine-tuned (full)",
    "lora": "Chronos-2 fine-tuned (LoRA)",
}


def _require_chronos() -> None:
    """Fail early with a useful message when Chronos is not installed."""
    if importlib.util.find_spec("chronos") is None:
        raise RuntimeError(
            "The Chronos package is not installed. Install the project "
            "dependencies first; fine-tuning additionally requires "
            "chronos-forecasting >= 2.2.0."
        )


def _read_daily_input(path: Path) -> list[float]:
    """Read exactly 24 finite, non-negative hourly consumption values.

    Values may be separated by whitespace, commas or semicolons. Lines may
    contain comments after '#'. The unit is kW, matching the project dataset.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input file was not found: {path}\n"
            f"Create {DEFAULT_INPUT_PATH.relative_to(PROJECT_ROOT)} and put "
            "24 hourly consumption values in it (one value per line is easiest)."
        )

    content_lines = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        content_lines.append(line.split("#", 1)[0])
    content = "\n".join(content_lines).strip()
    tokens = [token for token in re.split(r"[\s,;]+", content) if token]

    if not tokens:
        raise ValueError(f"Input file is empty: {path}")

    values: list[float] = []
    for index, token in enumerate(tokens, start=1):
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid number at position {index} in {path}: {token!r}"
            ) from exc
        if not (float("-inf") < value < float("inf")):
            raise ValueError(
                f"Input value #{index} is not finite: {token!r}"
            )
        if value < 0:
            raise ValueError(
                f"Input value #{index} is negative ({value}). "
                "Electricity consumption must be non-negative."
            )
        values.append(value)

    if len(values) != 24:
        raise ValueError(
            f"Expected exactly 24 hourly values in {path}, got {len(values)}."
        )

    return values


def _get_model_loader(model_key: str):
    """Return a loader for one of the three Chronos-2 variants."""
    _require_chronos()

    if model_key == "zero-shot":
        model_module = importlib.import_module("chronos2.load_model")
        return model_module.load_model

    tuned_module = importlib.import_module("chronos2.load_tunned_model")
    return lambda: tuned_module.load_model(mode=model_key)


def _check_checkpoints(model_keys: list[str]) -> None:
    """Validate local full/LoRA checkpoints before starting long operations."""
    if not any(key in {"full", "lora"} for key in model_keys):
        return

    _require_chronos()
    tuned_module = importlib.import_module("chronos2.load_tunned_model")
    missing = [
        f"{key}: {tuned_module.FINETUNED_MODEL_PATHS[key]}"
        for key in model_keys
        if key in {"full", "lora"}
        and not tuned_module.FINETUNED_MODEL_PATHS[key].exists()
    ]
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            "The requested fine-tuned checkpoint(s) do not exist:\n"
            f"{formatted}\nRun `python -m src.main train --model all` first."
        )


def _forecast_one(model_key: str, context: list[float]):
    """Load one model and predict the next 24 hourly values."""
    loader = _get_model_loader(model_key)
    model = loader()

    predict_module = importlib.import_module("chronos2.predict")
    prediction, quantiles = predict_module.predict_next_day(model, context)

    # Release model memory between model runs (particularly useful on GPU).
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return prediction, quantiles


def _write_forecast(
    output_path: Path,
    forecasts: dict[str, tuple[Any, Any]],
) -> None:
    """Write forecasts as a compact CSV-formatted text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp")

    with temp_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["model", "hour_ahead", "prediction_kw", "q10_kw", "q50_kw", "q90_kw"]
        )
        for model_key, (prediction, quantiles) in forecasts.items():
            for hour in range(24):
                writer.writerow(
                    [
                        MODEL_LABELS[model_key],
                        hour + 1,
                        f"{float(prediction[hour]):.6f}",
                        f"{float(quantiles[hour][0]):.6f}",
                        f"{float(quantiles[hour][1]):.6f}",
                        f"{float(quantiles[hour][2]):.6f}",
                    ]
                )

    temp_path.replace(output_path)


def run_input_forecast(
    model: str = "zero-shot",
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Forecast the next day from data/input.txt (or a supplied file)."""
    model_keys = list(MODEL_KEYS) if model == "all" else [model]
    if any(key not in MODEL_KEYS for key in model_keys):
        raise ValueError(f"Unknown model selection: {model}")

    _check_checkpoints(model_keys)
    context = _read_daily_input(input_path)

    print("=== ELECTRICITY FORECAST ===")
    print(f"Input: {input_path}")
    print(f"Input values: {len(context)} hourly measurements (kW)")
    print(
        "Adaptation: the 24 supplied measurements are used as the model's "
        "forecast context; model weights are not updated during prediction."
    )

    forecasts: dict[str, tuple[Any, Any]] = {}
    for model_key in model_keys:
        print(f"\nLoading {MODEL_LABELS[model_key]}...")
        prediction, quantiles = _forecast_one(model_key, context)
        forecasts[model_key] = (prediction, quantiles)

        print(f"\n--- {MODEL_LABELS[model_key]} ---")
        for hour, value in enumerate(prediction, start=1):
            print(f"Hour +{hour:02d}: {float(value):.3f} kW")

    _write_forecast(output_path, forecasts)
    print(f"\nSaved forecast: {output_path}")


def _patch_chronos_constants() -> None:
    """Expose shared forecasting constants where legacy modules expect them."""
    predict_module = importlib.import_module("chronos2.predict")
    base_model_module = importlib.import_module("chronos2.load_model")
    for constant_name in (
        "CONTEXT_LENGTH",
        "PREDICTION_LENGTH",
        "QUANTILE_LEVELS",
    ):
        if not hasattr(base_model_module, constant_name):
            setattr(
                base_model_module,
                constant_name,
                getattr(predict_module, constant_name),
            )


def train_models(
    model: str = "all",
    steps: int = 300,
    batch_size: int = 8,
    learning_rate: float | None = None,
) -> None:
    """Fine-tune the existing Chronos-2 model in full and/or LoRA mode.

    The zero-shot variant is the upstream pretrained model and is not trained
    locally. `all` trains both local fine-tuned variants.
    """
    if model not in {"all", "full", "lora"}:
        raise ValueError("Training model must be 'all', 'full' or 'lora'.")
    if steps <= 0:
        raise ValueError("--steps must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")

    _require_chronos()
    tuned_module = importlib.import_module("chronos2.load_tunned_model")
    modes = ["full", "lora"] if model == "all" else [model]

    print("=== CHRONOS-2 MODEL TRAINING ===")
    print("The zero-shot model is pretrained; local training creates full and LoRA variants.")

    # Bridge the constants' location mismatch without editing other modules.
    _patch_chronos_constants()

    for mode in modes:
        # finetune.py imports FINETUNED_MODEL_PATH, but the current loader only
        # defines FINETUNED_MODEL_PATHS. Supply the selected path here, without
        # editing the other project modules.
        tuned_module.FINETUNED_MODEL_PATH = tuned_module.FINETUNED_MODEL_PATHS[mode]
        finetune_module = importlib.import_module("chronos2.finetune")
        finetune_module.FINETUNED_MODEL_PATH = tuned_module.FINETUNED_MODEL_PATHS[mode]

        print(f"\n=== TRAINING {mode.upper()} ===")
        finetune_module.finetune(
            finetune_mode=mode,
            num_steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    print("\nTraining command completed.")


def _evaluate_one(model_key: str, evaluation_module) -> Path:
    """Run the existing evaluator for one model in its own output directory."""
    from pathlib import Path

    forecast_dir = PROJECT_ROOT / "results" / "evaluation" / model_key / "forecasts"
    plot_dir = PROJECT_ROOT / "results" / "evaluation" / model_key / "plots"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    old_forecasts_dir = evaluation_module.FORECASTS_DIR
    old_plots_dir = evaluation_module.PLOTS_DIR
    old_loader = evaluation_module.MODEL_LOADERS.get("finetuned")

    evaluation_module.FORECASTS_DIR = forecast_dir
    evaluation_module.PLOTS_DIR = plot_dir

    try:
        if model_key == "zero-shot":
            evaluation_module.evaluate(model_type="zero-shot")
            metrics_path = forecast_dir / "test_metrics_zero_shot.csv"
        else:
            tuned_module = importlib.import_module("chronos2.load_tunned_model")
            evaluation_module.MODEL_LOADERS["finetuned"] = (
                MODEL_LABELS[model_key],
                lambda mode=model_key: tuned_module.load_model(mode=mode),
            )
            evaluation_module.evaluate(model_type="finetuned")
            metrics_path = forecast_dir / "test_metrics_finetuned.csv"
    finally:
        evaluation_module.FORECASTS_DIR = old_forecasts_dir
        evaluation_module.PLOTS_DIR = old_plots_dir
        if old_loader is not None:
            evaluation_module.MODEL_LOADERS["finetuned"] = old_loader

    return metrics_path


def evaluate_models(model: str = "all") -> None:
    """Evaluate the selected variants on the common held-out test split."""
    aliases = {
        "finetuned": "full",  # compatibility with the previous main.py
        "both": "zero-shot,full",
    }
    if model in aliases:
        model_keys = aliases[model].split(",")
    elif model == "all":
        model_keys = list(MODEL_KEYS)
    elif model in MODEL_KEYS:
        model_keys = [model]
    else:
        raise ValueError("Evaluation model must be 'zero-shot', 'full', 'lora' or 'all'.")

    _check_checkpoints(model_keys)
    # evaluation.py imports the same constants from load_model.py.
    _patch_chronos_constants()
    evaluation_module = importlib.import_module("src.evaluation")

    comparison_dir = PROJECT_ROOT / "results" / "evaluation"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    metric_paths: dict[str, Path] = {}

    for model_key in model_keys:
        print(f"\n{'=' * 12} EVALUATING {MODEL_LABELS[model_key].upper()} {'=' * 12}")
        metric_paths[model_key] = _evaluate_one(model_key, evaluation_module)

    # Combine the existing evaluator's per-model metric files into one table.
    comparison_rows: list[dict[str, str]] = []
    baseline_row: dict[str, str] | None = None
    for model_key, metrics_path in metric_paths.items():
        with metrics_path.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            if row.get("model") == "Seasonal Naive":
                if baseline_row is None:
                    baseline_row = row
            else:
                comparison_rows.append(row)

    if baseline_row is not None:
        comparison_rows.append(baseline_row)

    comparison_path = comparison_dir / "model_comparison.csv"
    if comparison_rows:
        with comparison_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comparison_rows)

    print("\n=== MODEL COMPARISON ===")
    for row in comparison_rows:
        print(
            f"{row['model']:<34} "
            f"MAE={float(row['MAE_kW']):.4f} kW  "
            f"RMSE={float(row['RMSE_kW']):.4f} kW  "
            f"R2={float(row['R2']):.4f}"
        )
    if comparison_rows:
        print(f"\nSaved comparison: {comparison_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Household electricity forecasting with Chronos-2."
    )
    subparsers = parser.add_subparsers(dest="command")

    forecast_parser = subparsers.add_parser(
        "forecast",
        help="Forecast the next 24 hours from data/input.txt.",
    )
    forecast_parser.add_argument(
        "--model",
        choices=["zero-shot", "full", "lora", "all", "finetuned"],
        default="zero-shot",
        help="Model variant (default: zero-shot; 'finetuned' is an alias for 'full').",
    )
    forecast_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    forecast_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)

    train_parser = subparsers.add_parser(
        "train",
        help="Create the local full and/or LoRA fine-tuned Chronos-2 model.",
    )
    train_parser.add_argument("--model", choices=["all", "full", "lora"], default="all")
    train_parser.add_argument("--steps", type=int, default=300)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=None)

    # Keep the old command name working, while exposing the clearer `train`.
    finetune_parser = subparsers.add_parser(
        "finetune",
        help="Compatibility alias for training one fine-tuned variant.",
    )
    finetune_parser.add_argument("--mode", choices=["full", "lora"], default="full")
    finetune_parser.add_argument("--steps", type=int, default=300)
    finetune_parser.add_argument("--batch-size", type=int, default=8)
    finetune_parser.add_argument("--learning-rate", type=float, default=None)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate selected model(s) on the held-out test split.",
    )
    evaluate_parser.add_argument(
        "--model",
        choices=["zero-shot", "full", "lora", "all", "finetuned", "both"],
        default="all",
        help="Model to evaluate (default: all three variants).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # No subcommand means the simplest everyday workflow: read input.txt,
    # forecast the next day with zero-shot Chronos-2, and write output.txt.
    command = args.command or "forecast"
    try:
        if command == "forecast":
            raw_model = getattr(args, "model", "zero-shot")
            model = "full" if raw_model == "finetuned" else raw_model
            run_input_forecast(
                model=model,
                input_path=getattr(args, "input", DEFAULT_INPUT_PATH),
                output_path=getattr(args, "output", DEFAULT_OUTPUT_PATH),
            )
        elif command == "train":
            train_models(
                model=args.model,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
        elif command == "finetune":
            train_models(
                model=args.mode,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
        elif command == "evaluate":
            evaluate_models(model=args.model)
        else:
            parser.error(f"Unknown command: {command}")
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
