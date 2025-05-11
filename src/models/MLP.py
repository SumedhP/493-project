from typing import List
import torch
import torch.nn as nn
import lightning as L


class MLP(nn.Module):
    def __init__(self, layer_dims: List[int]):
        super().__init__()

        INPUT_DIM = 400 * 3

        dims = [INPUT_DIM] + layer_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten (N, 400, 3) -> (N, 1200)
        return self.model(x)


class MLPLightning(L.LightningModule):
    def __init__(self, layer_dims: List[int] = [256, 64], lr=1e-3, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()

        self.model = MLP(layer_dims)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_flattened = y.float().unsqueeze(1)
        loss = self.loss_fn(logits, y_flattened)
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y_flattened).float().mean()
        self.log_dict({"train_loss": loss, "train_acc": acc}, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_flattened = y.float().unsqueeze(1)
        loss = self.loss_fn(logits, y_flattened)
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y_flattened).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_flattened = y.float().unsqueeze(1)
        loss = self.loss_fn(logits, y_flattened)
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y_flattened).float().mean()
        self.log_dict({"test_loss": loss, "test_acc": acc}, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
