import pandas as pd
from dataloader.dataset import AccelDataset
from models.MLP import MLP
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

print("Starting training...")
df = pd.read_csv("data/small_dataset.csv")
dataset = AccelDataset(df)

print(f"The dataset has: {len(dataset)} points")
dataloader = DataLoader(dataset, batch_size=10)

model = MLP()
loss = nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 100
losses = []

print(f"Starting training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    epoch_loss = 0
    for i, (x, y) in enumerate(dataloader):
        optim.zero_grad()
        y_pred = model(x)
        loss_value = loss(y_pred, y.float().unsqueeze(1))
        loss_value.backward()
        optim.step()

        epoch_loss += loss_value.item()

        if i % 100 == 0:
            print(f"Epoch {epoch}, Batch {i}, Loss: {loss_value.item()}")

    avg_epoch_loss = epoch_loss / len(dataloader)
    losses.append(avg_epoch_loss)
    print(f"Epoch {epoch} completed. Avg Loss: {avg_epoch_loss:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(range(EPOCHS), losses, marker="o", label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
