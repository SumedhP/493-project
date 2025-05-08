import pandas as pd
from torch.utils.data import Dataset
import numpy as np


class AccelDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sliding_window_size: int = 10,
        sliding_window_stride: int = 1,
        hz: int = 40,
    ): ...
