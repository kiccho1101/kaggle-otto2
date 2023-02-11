from pathlib import Path

import pandas as pd
import polars as pl

from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import FeatureBase

"""
userの最初と最後のイベントの時間差
- session
- user_ts_diff
- user_min_all_min_ts_diff
- user_max_all_min_ts_diff
- user_min_all_max_ts_diff
- user_max_all_max_ts_diff
"""


class UserTsDiff(FeatureBase):
    def __init__(self, root_dir: Path, data_loader: OttoDataLoader):
        super().__init__(
            root_dir=root_dir,
            key_cols=["session"],
            data_loader=data_loader,
            fillna=-1,
            dtype=pl.Int32,
        )

    def _fit(self):
        test_df = self.data_loader.get_test_df().to_pandas()
        all_min_ts = test_df["ts"].min()
        all_max_ts = test_df["ts"].max()
        user_min_ts = test_df.groupby("session")["ts"].min().rename("user_min_ts")
        user_max_ts = test_df.groupby("session")["ts"].max().rename("user_max_ts")
        sessions = user_min_ts.index

        feat_df = pl.DataFrame(
            pd.DataFrame(
                {
                    "session": sessions,
                    "user_ts_diff": (user_max_ts - user_min_ts).abs(),
                    "user_min_all_min_ts_diff": (user_min_ts - all_min_ts).abs(),
                    "user_max_all_min_ts_diff": (user_max_ts - all_min_ts).abs(),
                    "user_min_all_max_ts_diff": (user_min_ts - all_max_ts).abs(),
                    "user_max_all_max_ts_diff": (user_max_ts - all_max_ts).abs(),
                }
            )
        ).fill_null(self.fillna)

        feat_df = feat_df.with_columns(
            [pl.col(col).cast(pl.Int32) for col in feat_df.columns if col != "session"]
        ).with_column(pl.col("session").cast(pl.Int32))
        return feat_df
