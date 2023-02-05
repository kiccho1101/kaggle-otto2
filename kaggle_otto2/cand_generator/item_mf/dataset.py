import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class OttoPairDataset(Dataset):
    def __init__(self, pair_df: pd.DataFrame, train_df: pd.DataFrame, aid2idx):
        super().__init__()
        self.aid_x = pair_df["aid_idx"].tolist()
        self.aid_y = pair_df["next_aid_idx"].tolist()
        self.aid_size = train_df.groupby("aid").size()
        self.aid_size.index = self.aid_size.index.map(aid2idx)
        self.ts_diff = pair_df["ts_diff"].tolist()
        self.max_ts_diff = 7 * 24 * 60 * 60  # 7 days
        self.max_aid_size = self.aid_size.max()

    def __len__(self):
        return len(self.aid_x)

    def __getitem__(self, index: int):
        aid_x = self.aid_x[index]
        aid_y = self.aid_y[index]
        size_x = self.aid_size[aid_x]
        size_y = self.aid_size[aid_y]
        ts_diff = self.ts_diff[index]
        ts_diff_coef = np.log1p(min(self.max_ts_diff, ts_diff) / self.max_ts_diff)
        return {
            "aid_x": aid_x,
            "aid_y": aid_y,
            "size_x": 1 / size_x / (ts_diff_coef + 1),
            "size_y": 1 / size_y,
        }
