from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "dataset" / "household_power_consumption.txt"
OUTPUT_DIR = ROOT_DIR / "results" / "plots"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        na_values=["?"],
        low_memory=False
    )

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S"
    )

    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"],
        errors="coerce"
    )

    df = df.set_index("datetime").sort_index()

    return df


def main():
    df = load_data()

    power = df["Global_active_power"]

    hourly = power.resample("1h").agg(["mean", "count"])
    hourly.columns = ["power", "valid_minutes"]

    hourly.to_csv(PROCESSED_DIR / "hourly_power_raw.csv")

if __name__ == "__main__":
    main()
