import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class AccelDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sliding_window_size: int = 10,
        sliding_window_stride: int = 1,
        hz: int = 40,
    ):

        self.data = []
        self.samples_per_window = sliding_window_size * hz
        self.stride = sliding_window_stride * hz

        for pid, group in df.groupby("pid"):
            # Ensure the data is sorted timewise
            group = group.sort_values("time").reset_index(drop=True)

            X = group[["x", "y", "z"]].to_numpy(dtype=np.float32)
            y = group["intoxicated"].to_numpy(dtype=np.uint32)

            # Take slices over the data
            for start_index in range(0, len(group) - self.samples_per_window + 1, self.stride):
                end_index = start_index + self.samples_per_window
                window_x = X[start_index:end_index]
                window_y = y[start_index:end_index]

                # If the window of y contains any datapoint that's true, the whole datapoint is true
                label = int(window_y.max())
                self.data.append((window_x, label))

        labels = [label for _, label in self.data]
        label_counts = np.bincount(labels)
        print(f"Created dataset. Label counts: {label_counts}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.uint32)
        return x, y


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import time

    print(f"Started at {time.time()}")
    start_time = time.time()
    df = pd.read_csv("../data/combined_data.csv")
    print(f"Took {time.time() - start_time} seconds to read in the csv")
    print(f"The csv data has: {len(df)} points")
    print("Sample values:")
    print(df[:100])


    start_time = time.time()
    dataset = AccelDataset(df)
    print(f"Took {time.time() - start_time} seconds to make the dataset")
    print(f"The dataset has: {len(dataset)} points")

    dataloader = DataLoader(dataset)

    for x, y in dataloader:
        print(x)
        print(x.shape)
        print(y.shape)
        print(y)
        break
