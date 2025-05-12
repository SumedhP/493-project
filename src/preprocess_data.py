import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pids = [
    "BK7610",
    "BU4707",
    "CC6740",
    "DC6359",
    "DK3500",
    "HV0618",
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


def clean_up_data(pid_acc_data : pd.DataFrame, pid_tac_data : pd.DataFrame):
    """
    This should remove any duplicate timestamps from the accelerometer data
    Then, clip the acc_data to only be from beginning and and of the tac data
    Then, look for continuous segments of data in the acc data, where the gap between 1 datapoint and the next is less than
    0.5s. If this segment is greater than 10s than add to a running list.
    Then, in that data, interpolate 40hz data 
    Then, join against tac data
    """
    df_acc = pid_acc_data.drop_duplicates(subset="time").sort_values("time").copy()
    times_ms = df_acc["time"].values
    
    df_tac = pid_tac_data.copy()
    df_tac["timestamp"] = (df_tac["timestamp"] * 1000).astype(np.int64) # Convert to ms
    start_time, end_time = df_tac["timestamp"].min(), df_tac["timestamp"].max()
    
    time_diff = np.diff(df_acc["time"]) / 1000.0 # Get gaps in s
    gaps = np.where(time_diff >= 1)[0] # Get where theres gaps in the data
    boundaries = np.concatenate(([0], gaps + 1, [len(df_acc)]))
    segments = []
    
    for i in range(len(boundaries) - 1):
        seg = df_acc.iloc[boundaries[i]:boundaries[i+1]] # Get segment
        duration = (seg["time"].iloc[-1] - seg["time"].iloc[0]) / 1000.0 # Get duration in s
        if duration >= 10.0:
            # Only accept long segments
            segments.append(seg)
    
    
    if not segments:
        raise Exception("This person has no useable segments")
    
    # 4. Resample each segment to 40Hz
    resampled_list = []
    for segment in segments:
        segment = segment.set_index("time")[['x','y','z']]
        t0, t1 = segment.index.min(), segment.index.max()
        # Align to next full second and previous full second
        start_ms = int(np.ceil(t0 / 1000.0) * 1000)
        end_ms = int(np.floor(t1 / 1000.0) * 1000)
        if end_ms <= start_ms:
            continue
        idx = np.arange(segment.index.min(), segment.index.max(), 25)
        resample_segment = (segment.reindex(idx)
                  .interpolate()
                  .ffill()
                  .bfill()
                 )
        resample_segment = resample_segment.reset_index().rename(columns={"index": "time"})
        resampled_list.append(resample_segment)

    resampled = pd.concat(resampled_list, ignore_index=True)
    resampled["pid"] = pid_acc_data["pid"].iat[0]
    resampled["time"] = resampled["time"].round().astype(np.int64)

    # Merge with TAC data
    merged = pd.merge_asof(
        resampled.sort_values("time"),
        df_tac.sort_values("timestamp"),
        left_on="time", right_on="timestamp",
        direction="backward"
    )
    
    # 6. Setup intoxication boolean
    merged["intoxicated"] = (merged["TAC_Reading"] > 0.08).astype(int)
    return merged[["time","pid","x","y","z","intoxicated"]]


all_combined = []

for pid in pids:
    print()
    print(f"Looking at pid {pid}")
    
    tac_file = f"data/clean_tac/{pid}_clean_TAC.csv"
    tac_data = pd.read_csv(tac_file)

    pid_acc_data = acc_data[acc_data["pid"] == pid].copy()
    cleaned = clean_up_data(pid_acc_data, tac_data)
    print("Created dataset with datapoints: ", len(cleaned))
    all_combined.append(cleaned)

final_df = pd.concat(all_combined, ignore_index=True)
print("Created BIG DATASET with datapoints: ", len(final_df))

print(final_df)
print(final_df["intoxicated"].mean())

final_df.to_csv("combined_dataset.csv", index=False)

small_dataset = final_df[:1_000_000].copy()
small_dataset.to_csv("small_dataset.csv", index=False)

