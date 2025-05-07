import pandas as pd

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
    # print("Starting time: ", min(time))
    # print("Ending time: ", max(time))
    # print("Duration (ms): ", max(time) - min(time))
    
    # print("Overall have datapoints: ", len(time))
    # print("Expected at 40hz: ", (max(time) - min(time)) / 1000 * 40)
    
    tac_data["timestamp"] = tac_data["timestamp"] * 1000
    
    # tac_time, tac_bac = tac_data["timestamp"], tac_data["TAC_reading"]
    # tac_time *= 1000 # Converts from s to ms to match accel data
    
    # LEFT OUTER JOIN TIME
    merged = pd.merge_asof(
        pid_acc_data,
        tac_data,
        left_on="time",
        right_on="timestamp",
        direction="backward",
        suffixes=("", "_tac")
    )
    
    merged["intoxicated"] = (merged["TAC_Reading"] > 0.08).astype(int)
    # print(merged["intoxicated"].mean())
    # print(merged)
    
    cleaned = merged[["time", "pid", "x", "y", "z", "intoxicated"]]
    
    all_combined.append(cleaned)
    
final_df = pd.concat(all_combined, ignore_index=True)
final_df.to_csv("combined_data.csv", index=False)

print(final_df)
print(final_df["intoxicated"].mean())

