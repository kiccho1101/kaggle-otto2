import numpy as np
import polars as pl
from torch.utils.data import Dataset


class UserMFDataset(Dataset):
    def __init__(self, train_df: pl.DataFrame, session2idx, aid2idx):
        super().__init__()
        train_df = train_df.with_columns(
            [
                pl.col("session")
                .apply(lambda x: session2idx[x])
                .alias("session_idx")
                .cast(pl.Int32),
                pl.col("aid")
                .apply(lambda x: aid2idx[x])
                .alias("aid_idx")
                .cast(pl.Int32),
            ]
        )
        self.sessions = train_df["session_idx"].to_numpy()
        self.aids = train_df["aid_idx"].to_numpy()
        aid_vc = train_df["aid_idx"].value_counts()
        self.aid_size = dict(
            zip(aid_vc["aid_idx"].to_list(), aid_vc["counts"].to_list())
        )

        self.max_ts_diff = 7 * 24 * 60 * 60  # 7 days

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, index: int):
        session = self.sessions[index]
        aid = self.aids[index]
        aid_size = self.aid_size[aid]
        return {
            "session": session,
            "aid": aid,
            "aid_size": 1 / np.log1p(aid_size),
        }
