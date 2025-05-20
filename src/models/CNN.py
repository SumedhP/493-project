import torch
import torch.nn as nn
import lightning as L


class CNN(nn.Module):
    def __init__(
        self,
        conv_channels: list[int] = [64, 128, 256],
    ):
        super().__init__()

        # Initial conv block
        # self.feature_extractor = nn.Sequential(
        #     nn.Conv1d(in_channels=3, out_channels=conv_out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(conv_out_channels),
        #     nn.LeakyReLU(),
        #     nn.Dropout(dropout_prob),
        # )

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=conv_out_channels, kernel_size=6),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(conv_out_channels)
        )

        conv_output_length = 77

        # # Pass through a fake input to figure out the output size
        dummy_input = torch.zeros(1, 3, 400)
        flat_size = self.feature_extractor(dummy_input).view(1, -1).size(1)

        # # Make it all into one now
        # self.model = nn.Sequential(self.feature_extractor, nn.Flatten(), nn.Linear(flat_size, 1))
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 1028),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1028, 1)  # Output: 2 classes (intoxicated, sober)
        )
        # test_acc: 0.6380000114440918


    def forward(self, x):
        x = x.permute(0, 2, 1)  # (N, 400, 3) -> (N, 3, 400) to make it fit input proper
        # return self.model(x)

        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x


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
