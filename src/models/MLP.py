import torch
import torch.nn as nn
import lightning as L


class MLP(nn.Module):
    def __init__(self, lr=1e-3):
        super().__init__()

        INPUT_DIM = 400 * 3

        self.model = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten (N, 400, 3) -> (N, 1200)
        return self.model(x)


class MLPLightning(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = MLP(lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_flattened = y.float().unsqueeze(1)
        loss = self.loss_fn(logits, y_flattened)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
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
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
