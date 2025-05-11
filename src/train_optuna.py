import pandas as pd
from dataloader.dataset import AccelDataLightning
from models.MLP import MLPLightning
import lightning as L
import time
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

DF_ORIG = None


def objective(trial: optuna.trial.Trial) -> float:
    df = DF_ORIG.copy()

    # Suggest hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layer_dims = []
    for i in range(n_layers):
        dim = trial.suggest_int(f"layer_{i+1}_dim", 32, 512, log=True)
        layer_dims.append(dim)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)

    hyperparameters = dict(
        n_layers=n_layers,
        layer_dims=layer_dims,
        lr=lr,
        weight_decay=weight_decay,
    )

    # DataModule (uses df copy)
    datamodule = AccelDataLightning(df, sliding_window_stride=10, batch_size=10)

    # Model
    model = MLPLightning(layer_dims, lr, weight_decay)

    # Callbacks: early stopping + pruning

    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    pruning = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    trainer = L.Trainer(
        max_epochs=50,
        callbacks=[early_stop, pruning],
        enable_progress_bar=False,
        logger=False,
    )

    # Fit and return validation loss
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)
    return trainer.callback_metrics["val_acc"].item()


def main():
    print("Loading in data...")
    df = pd.read_csv("data/medium_dataset.csv")

    global DF_ORIG
    DF_ORIG = df

    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
    )
    
    print("Starting hyperparameter optimization...")
    study.optimize(
        objective,
        n_trials=100,
        timeout=600,  # 10 minutes
        show_progress_bar=True,
        n_jobs=2
    )
    
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.value}")
    
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()  # This iss needed for lightning not to crash on Windows
    main()
