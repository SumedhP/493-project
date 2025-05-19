import torch
import torch.nn as nn
import lightning as L


class CNN(nn.Module):
    def __init__(
        self,
        conv_channels: list[int] = [64, 128, 256],
    ):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        in_ch = 3
        for out_ch in conv_channels:
            block = nn.Sequential(
                nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(),
            )
            self.conv_blocks.append(block)
            in_ch = out_ch

        # Calculate flattened size
        dummy = torch.zeros(1, 3, 400)
        with torch.no_grad():
            for block in self.conv_blocks:
                dummy = block(dummy)
        flat_size = dummy.view(1, -1).size(1)

        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, channels)
        x = x.permute(0, 2, 1)  # to (batch, channels, seq_len)
        for block in self.conv_blocks:
            x = block(x)
        logits = self.classifier(x)
        return logits


class CNNLightning(L.LightningModule):
    def __init__(
        self,
        conv_channels: list[int] = [64, 128, 256, 1024],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = CNN(
            conv_channels=conv_channels
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.float().unsqueeze(1)
        loss = self.loss_fn(logits, y)
        acc = ((torch.sigmoid(logits) > 0.5) == y).float().mean()
        self.log_dict({"train_loss": loss, "train_acc": acc}, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.float().unsqueeze(1)
        loss = self.loss_fn(logits, y)
        acc = ((torch.sigmoid(logits) > 0.5) == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.float().unsqueeze(1)
        loss = self.loss_fn(logits, y)
        acc = ((torch.sigmoid(logits) > 0.5) == y).float().mean()
        self.log_dict({"test_loss": loss, "test_acc": acc})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
