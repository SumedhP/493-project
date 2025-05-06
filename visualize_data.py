import matplotlib.pyplot as plt
import pandas as pd

# First plot: TAC Reading over time
file = "src/data/clean_tac/BK7610_clean_TAC.csv"
data = pd.read_csv(file)
print(data)

tac_time = data["timestamp"]
print("Data start point timestamp: ", min(tac_time))
print("Data end point timestamp: ", max(tac_time))

# tac_time -= min(tac_time)

print("Data duration data point timestamp: ", max(tac_time))

bac = data["TAC_Reading"]

plt.figure(figsize=(10, 4))
plt.plot(tac_time, bac, label="TAC Reading")
plt.xlabel("Timestamp")
plt.ylabel("TAC Reading")
plt.title("TAC Reading Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
# plt.show()

# Second plot: Accelerometer data (x, y, z) over time for BK7610
acc_data_file = "src/data/all_accelerometer_data_pids_13.csv"
acc_data = pd.read_csv(acc_data_file)

bk7_data = acc_data[acc_data["pid"] == "BK7610"]
acc_time = bk7_data["time"]
acc_time /= 1000

print("Accelerometer data:")
print("Data start point timestamp: ", min(acc_time))
print("Data end point timestamp: ", max(acc_time))

# acc_time -= min(acc_time)

x, y, z = bk7_data["x"], bk7_data["y"], bk7_data["z"]

print("Data duration: ", max(acc_time))

min_acc_time = min(acc_time)
max_acc_time = max(acc_time)

tac_time_mask = (tac_time >= min_acc_time) & (tac_time <= max_acc_time)
data = data[tac_time_mask]
tac_time = data["timestamp"]
bac = data["TAC_Reading"] * 100

print("There are ", len(bac), " datapoints in this duration")


plt.figure(figsize=(10, 4))
plt.plot(acc_time, x, label="X")
plt.plot(acc_time, y, label="Y")
plt.plot(acc_time, z, label="Z")
plt.plot(tac_time, bac, label="BAC")
plt.xlabel("Time")
plt.ylabel("Acceleration")
plt.title("Accelerometer Data for BK7610")
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.show()
