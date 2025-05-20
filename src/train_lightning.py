import pandas as pd
from dataloader.dataset import AccelDataLightning
from models.MLP import MLPLightning
from models.CNN import CNNLightning
import lightning as L
import time


def main():

    print("Loading in data...")
    # df = pd.read_csv("data/small_dataset.csv")
    df = pd.read_csv("src/data/small_dataset.csv")
    # df = pd.read_csv("data/combined_data.csv")
    dataset = AccelDataLightning(df, sliding_window_stride=10, batch_size=16)

    # model = MLPLightning()
    model = CNNLightning()
    trainer = L.Trainer(max_epochs=100)

    start_time = time.time()
    print("Starting training...")
    trainer.fit(model, datamodule=dataset)
    print("Finished training.")
    print("Starting testing...")
    trainer.test(model, datamodule=dataset)
    print("Finished testing.")

    print(f"Total time spent training: {time.time() - start_time} seconds")
    print("Finished.")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()  # This iss needed for lightning not to crash on Windows
    main()
