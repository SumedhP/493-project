import pandas as pd
from dataloader.dataset import AccelDataLightning
from models.MLP import MLPLightning
import lightning as L
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

DF_ORIG = None


def objective(trial: optuna.trial.Trial) -> float:
    df = DF_ORIG.copy()

    # Suggest hyperparameters
    n_layers = 5
    layer_dims = []
    for i in range(n_layers):
        dim = trial.suggest_int(f"layer_{i+1}_dim", 32, 200, log=True)
        layer_dims.append(dim)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    hyperparameters = dict(
        layer_dims=layer_dims,
        weight_decay=weight_decay,
    )

    # DataModule (uses df copy)
    datamodule = AccelDataLightning(df, sliding_window_stride=1, batch_size=64)

    # Model
    model = MLPLightning(layer_dims, weight_decay=weight_decay)

    # Callbacks: early stopping + pruning

    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    pruning = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    tensorboard_logger = TensorBoardLogger(save_dir="optuna_logs", name="optuna_logs", default_hp_metric=False)
    trainer = L.Trainer(
        max_epochs=50,
        callbacks=[early_stop, pruning],
        enable_progress_bar=False,
        logger=tensorboard_logger,
        enable_model_summary=False,
    )

    # Fit and return validation loss
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)
    return trainer.callback_metrics["val_acc"].item()


def main():
    print("Loading in data...")
    # df = pd.read_csv("data/combined_data.csv")
    df = pd.read_csv("data/small_dataset.csv")

    global DF_ORIG
    DF_ORIG = df

    lock_obj = optuna.storages.journal.JournalFileOpenLock("./optuna_journal_storage.log")
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend("./optuna_journal_storage.log", lock_obj=lock_obj),
    )
    
    study = optuna.create_study(
        storage=storage,
        direction="maximize"
    )
    
    print("Starting hyperparameter optimization...")
    study.optimize(
        objective,
        n_trials=1000,
        timeout=15 * 60 * 1,  # 15 minutes
        show_progress_bar=True,
        n_jobs=3
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

"""
MLP:
Number of finished trials: 35
Best trial: 0.6722532510757446
Best hyperparameters:
  n_layers: 3
  layer_1_dim: 34
  layer_2_dim: 36
  layer_3_dim: 194
  lr: 0.0009945382033371669
  weight_decay: 0.000556120150912317
  
n_layers 5
layer_1_dim 31
layer_2_dim 128
layer_3_dim 41
layer_4_dim 174
layer_5_dim 41
weight_decay 9.649623360646725e-05

Number of finished trials: 61
Best trial: 0.8051220774650574
Best hyperparameters:
  layer_1_dim: 156
  layer_2_dim: 198
  layer_3_dim: 59
  layer_4_dim: 51
  layer_5_dim: 33
  weight_decay: 1.0631590898739607e-05
"""
