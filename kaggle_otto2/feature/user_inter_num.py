from pathlib import Path

import polars as pl

from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import FeatureBase

"""
userのinter数
- session
- user_inter_num
- user_click_num
- user_cart_num
- user_order_num
"""


class UserInterNum(FeatureBase):
    def __init__(self, root_dir: Path, data_loader: OttoDataLoader):
        super().__init__(
            root_dir=root_dir,
            key_cols=["session"],
            data_loader=data_loader,
            fillna=-1,
            dtype=pl.Int32,
        )

    def _fit(self):
        test_df = self.data_loader.get_test_df()

        feat_dfs = []
        for t, event_name in self.event_types:
            feat_dfs.append(
                test_df.filter(pl.col("type").is_in(t))
                .groupby("session")
                .agg(pl.count("aid").cast(pl.Int32).alias(f"user_{event_name}_num"))
            )

        feat_df = feat_dfs[0]
        for df in feat_dfs[1:]:
            feat_df = feat_df.join(df, on="session", how="left")

        feat_df = feat_df.with_columns(
            [pl.col(col).cast(pl.Int32) for col in feat_df.columns if col != "session"]
        )
        return feat_df
