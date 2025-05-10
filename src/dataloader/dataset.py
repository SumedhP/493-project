import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
import lightning as L


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
            for start_index in range(
                0, len(group) - self.samples_per_window + 1, self.stride
            ):
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


class AccelDataLightning(L.LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        sliding_window_size: int = 10,
        sliding_window_stride: int = 1,
        hz: int = 40,
        batch_size: int = 32,
    ):
        super().__init__()
        self.data = AccelDataset(
            df,
            sliding_window_size=sliding_window_size,
            sliding_window_stride=sliding_window_stride,
            hz=hz,
        )
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Split into train (60%), val (20%), test (20%)
        train_size = int(0.6 * len(self.data))
        val_size = int(0.2 * len(self.data))
        test_size = len(self.data) - train_size - val_size
        self.train_data, self.val_data, self.test_data = random_split(
            self.data, [train_size, val_size, test_size]
        )
        print(
            f"Train size: {len(self.train_data)}, Val size: {len(self.val_data)}, Test size: {len(self.test_data)}"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import time

    print(f"Started at {time.time()}")
    start_time = time.time()
    df = pd.read_csv("/data/combined_data.csv")
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

    # Write first 1000 rows of the dataset to a csv file
    small_df = pd.DataFrame(df[:4000])
    small_df.to_csv("../data/small_dataset.csv", index=False)
    print("Finished")
