import argparse

from src.evaluation import evaluate
from src.predictor import run_forecast
from src.chronos2.finetune import finetune


def main():
    parser = argparse.ArgumentParser(
        description="Household electricity forecasting with Chronos-2"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model(s) on the held-out test split",
    )
    evaluate_parser.add_argument(
        "--model",
        choices=["zero-shot", "finetuned", "both"],
        default="zero-shot",
        help="Which Chronos-2 variant to evaluate (default: zero-shot)",
    )

    forecast_parser = subparsers.add_parser(
        "forecast",
        help="Forecast the next 24 hours from the latest available context",
    )
    forecast_parser.add_argument(
        "--model",
        choices=["zero-shot", "finetuned"],
        default="zero-shot",
        help="Which Chronos-2 variant to use (default: zero-shot)",
    )

    finetune_parser = subparsers.add_parser(
        "finetune",
        help="Fine-tune Chronos-2 using the training split",
    )
    finetune_parser.add_argument(
        "--mode",
        choices=["lora", "full"],
        default="lora",
        help="Fine-tuning strategy (default: lora)",
    )
    finetune_parser.add_argument("--steps", type=int, default=300)
    finetune_parser.add_argument("--batch-size", type=int, default=8)
    finetune_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Default: 1e-5 for LoRA, 1e-6 for full fine-tuning",
    )

    args = parser.parse_args()

    if args.command == "evaluate":
        evaluate(model_type=args.model)
    elif args.command == "forecast":
        run_forecast(model_type=args.model)
    elif args.command == "finetune":
        finetune(
            finetune_mode=args.mode,
            num_steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )


if __name__ == "__main__":
    main()
