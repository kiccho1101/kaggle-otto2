import numpy as np
from torch.utils.data import Dataset

from kaggle_otto2.data_loader import OttoDataLoader


class ItemMFDataset(Dataset):
    def __init__(self, data_loader: OttoDataLoader):
        super().__init__()
        pair_df = data_loader.get_pair_df()
        train_df = data_loader.get_train_df()
        train_df = train_df.join(data_loader.get_aid_idx_df(), on="aid")
        aid_vc = train_df["aid_idx"].value_counts()
        self.aid_size = dict(
            zip(aid_vc["aid_idx"].to_list(), aid_vc["counts"].to_list())
        )

        self.aid_x = pair_df["aid_idx"].to_numpy()
        self.aid_y = pair_df["next_aid_idx"].to_numpy()
        self.ts_diff = pair_df["ts_diff"].to_numpy()

        self.max_ts_diff = 7 * 24 * 60 * 60  # 7 days

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
            "coef_x": 1 / size_x / (ts_diff_coef + 1),
            "coef_y": 1 / size_y,
        }
