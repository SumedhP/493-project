import pandas as pd
from dataloader.dataset import AccelDataLightning
from models.MLP import MLPLightning
import lightning as L
import time

def main():

    print("Starting training...")
    df = pd.read_csv("data/medium_dataset.csv")
    dataset = AccelDataLightning(df)

    model = MLPLightning()
    trainer = L.Trainer(max_epochs=100)

    start_time = time.time()

    trainer.fit(model, datamodule=dataset)
    print("Finished training.")
    print("Starting testing...")
    trainer.test(model, datamodule=dataset)
    print("Finished testing.")

    print(f"Total time spent training: {time.time() - start_time} seconds")
    print("Finished.")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support() # This iss needed for lightning not to crash on Windows
    main()
