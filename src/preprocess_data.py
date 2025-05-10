import pandas as pd
import numpy as np

pids = [
    "BK7610",
    "BU4707",
    "CC6740",
    "DC6359",
    "DK3500",
    "HV0618",
    "JB3156",
    "JR8022",
    "MC7070",
    "MJ8002",
    "PC6771",
    "SA0297",
    "SF3079",
]

ACCEL_DATA_FILE = "data/all_accelerometer_data_pids_13.csv"
OUTPUT_DATA_FILE = "data/clean_combined_data.csv"
acc_data = pd.read_csv(ACCEL_DATA_FILE)

all_combined = []

for pid in pids:
    print()
    print(f"Looking at pid {pid}")
    PID_TAC_FILE = f"data/clean_tac/{pid}_clean_TAC.csv"
    tac_data = pd.read_csv(PID_TAC_FILE)

    """
    Example data:
    1493737156832,BK7610,-0.1353,0.2467,0.1022
    1493737156859,BK7610,-0.0811,0.3475,0.162
    """

    pid_acc_data = acc_data[acc_data["pid"] == pid]
    time = pid_acc_data["time"]

    # Get rid of duplicate timestamp values
    dupes = pid_acc_data["time"].duplicated().sum()
    print(f"PID {pid} has {dupes} exact-duplicate timestamps")
    pid_acc_data = pid_acc_data.drop_duplicates("time")

    modified_acc_data = pid_acc_data.set_index("time")
    modified_acc_data = modified_acc_data.drop(columns=["pid"])

    new_index = np.arange(pid_acc_data["time"].min(), pid_acc_data["time"].max(), 25)

    resampled_acc_data = (
        modified_acc_data.reindex(new_index)
        .infer_objects(copy=False)
        .interpolate()  # Make up data wahoo
        .ffill()  # carries first real value backward to start
        .bfill()  # carries last real value forward to end
    )
    resampled_acc_data = resampled_acc_data.reset_index().rename(
        columns={"index": "time"}
    )
    resampled_acc_data["pid"] = pid

    resampled_acc_data["second"] = resampled_acc_data["time"] // 1000
    counts = resampled_acc_data.groupby("second").size()
    valid_seconds = counts[counts == 40].index
    resampled_acc_data = resampled_acc_data[
        resampled_acc_data["second"].isin(valid_seconds)
    ].drop(columns=["second"])

    tac_data["timestamp"] = tac_data["timestamp"] * 1000

    # LEFT OUTER JOIN TIME
    merged = pd.merge_asof(
        resampled_acc_data,
        tac_data,
        left_on="time",
        right_on="timestamp",
        direction="backward",
        suffixes=("", "_tac"),
    )

    merged["intoxicated"] = (merged["TAC_Reading"] > 0.08).astype(int)

    cleaned = merged[["time", "pid", "x", "y", "z", "intoxicated"]].copy()

    all_combined.append(cleaned)

final_df = pd.concat(all_combined, ignore_index=True)
final_df.to_csv("combined_data.csv", index=False)

print(final_df)
print(final_df["intoxicated"].mean())
