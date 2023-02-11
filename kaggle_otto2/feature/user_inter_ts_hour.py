from pathlib import Path

import pandas as pd
import polars as pl

from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import FeatureBase

"""
userのinterの時間帯
- user_inter_ts_hour_mean
- user_click_ts_hour_mean
- user_cart_ts_hour_mean
- user_order_ts_hour_mean
"""


class UserInterTsHour(FeatureBase):
    def __init__(self, root_dir: Path, data_loader: OttoDataLoader):
        super().__init__(
            root_dir=root_dir,
            key_cols=["session"],
            data_loader=data_loader,
            fillna=-1,
            dtype=pl.Int8,
        )

    def _fit(self):
        test_df = self.data_loader.get_test_df().to_pandas()
        test_df["ts_dt"] = pd.to_datetime(test_df["ts"], unit="s")
        test_df["ts_hour"] = test_df["ts_dt"].dt.hour

        test_df_pl = pl.DataFrame(test_df)

        feat_dfs = []
        for t, event_name in self.event_types:
            feat_dfs.append(
                test_df_pl.groupby("session").agg(
                    [
                        pl.mean("ts_hour").alias(f"user_{event_name}_ts_hour_mean"),
                    ]
                )
            )
        feat_df = feat_dfs[0]
        for df in feat_dfs[1:]:
            feat_df = feat_df.join(df, on="session", how="left")

        feat_df = feat_df.with_columns(
            [pl.col(col).cast(pl.Int8) for col in feat_df.columns if col != "session"]
        )
        return feat_df
