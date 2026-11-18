#create .npz with (X, y) pairs

from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT_DIR / "data" / "processed" / "hourly_power_raw.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "forecasting_dataset.npz"

HOURS_PER_DAY = 24
MIN_VALID_MINUTES = 15

TRAIN_SHARE = 0.7
VAL_SHARE = 0.15
TEST_SHARE = 0.15

def load_data():
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["datetime"],
        index_col="datetime"
    )

    df = df.sort_index().asfreq("1h") #create empty rows if datetimes have gaps

    power = df["power"].fillna(0) #create 0 value if empty
    valid_minutes = df["valid_minutes"].fillna(0) #create 0 value if empty

    #check for valid minutes
    power[valid_minutes < MIN_VALID_MINUTES] = np.nan

    return power


def create_daily_pairs(series):
    X = []
    y = []

    values = series.to_numpy(dtype=np.float32)

    #find NaN
    valid = np.isfinite(values)
    
    #convert given binnaries into intervals
    edges = np.diff(
        np.concatenate(([False], valid, [False])).astype(np.int8)
    )

    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    for start, end in zip(starts, ends):

        #get sequential days
        segment = values[start:end]
        complete_days = len(segment) // HOURS_PER_DAY

        #creating pairs
        for day in range(complete_days - 1):
            input_start = day * HOURS_PER_DAY
            target_start = input_start + HOURS_PER_DAY

            X.append(
                segment[input_start:input_start + HOURS_PER_DAY]
            )

            y.append(
                segment[target_start:target_start + HOURS_PER_DAY]
            )

    return (
        np.asarray(X, dtype=np.float32).reshape(-1, HOURS_PER_DAY),
        np.asarray(y, dtype=np.float32).reshape(-1, HOURS_PER_DAY)
    )


def main():
    power = load_data()
    X, y = create_daily_pairs(power)

    #shuffle pairs using a fixed random seed
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(X))

    X = X[indices]
    y = y[indices]

    n = len(X)

    train_end = int(n * TRAIN_SHARE)
    val_end = int(n * (TRAIN_SHARE + VAL_SHARE))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    np.savez_compressed(
        OUTPUT_PATH,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test
    )

    print(f"Total days observed: {n+1}")
    print(f"Total hours observed: {(n+1)*HOURS_PER_DAY}")

    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation: X={X_val.shape}, y={y_val.shape}")
    print(f"Test: X={X_test.shape}, y={y_test.shape}")

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
