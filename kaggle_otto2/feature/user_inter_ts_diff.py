from pathlib import Path

import polars as pl

from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import FeatureBase

"""
userのinterの時間差
- session
- user_inter_ts_diff_min
- user_inter_ts_diff_mean
- user_inter_ts_diff_std
- user_inter_ts_diff_max
- user_click_ts_diff_min
- user_click_ts_diff_mean
- user_click_ts_diff_std
- user_click_ts_diff_max
- user_cart_ts_diff_min
- user_cart_ts_diff_mean
- user_cart_ts_diff_std
- user_cart_ts_diff_max
- user_order_ts_diff_min
- user_order_ts_diff_mean
- user_order_ts_diff_std
- user_order_ts_diff_max
"""


class UserInterTsDiff(FeatureBase):
    def __init__(self, root_dir: Path, data_loader: OttoDataLoader):
        super().__init__(
            root_dir=root_dir,
            key_cols=["session"],
            data_loader=data_loader,
            fillna=-1,
            dtype=pl.Float32,
        )

    def _fit(self):
        test_df = self.data_loader.get_test_df()

        feat_dfs = []
        for t, event_name in self.event_types:
            feat_dfs.append(
                test_df.filter(pl.col("type").is_in(t))
                .with_column(
                    (pl.col("ts").shift(-1).over("session") - pl.col("ts")).alias(
                        "ts_diff"
                    )
                )
                .groupby("session")
                .agg(
                    [
                        pl.min("ts_diff")
                        .cast(pl.Float32)
                        .alias(f"user_{event_name}_ts_diff_min"),
                        pl.mean("ts_diff")
                        .cast(pl.Float32)
                        .alias(f"user_{event_name}_ts_diff_mean"),
                        pl.std("ts_diff")
                        .cast(pl.Float32)
                        .alias(f"user_{event_name}_ts_diff_std"),
                        pl.max("ts_diff")
                        .cast(pl.Float32)
                        .alias(f"user_{event_name}_ts_diff_max"),
                    ]
                )
                .fill_null(self.fillna)
                .fill_nan(self.fillna)
            )

        feat_df = feat_dfs[0]
        for df in feat_dfs[1:]:
            feat_df = feat_df.join(df, on="session", how="left")

        feat_df = feat_df.with_columns(
            [
                pl.col(col).cast(pl.Float32)
                for col in feat_df.columns
                if col != "session"
            ]
        )
        return feat_df
