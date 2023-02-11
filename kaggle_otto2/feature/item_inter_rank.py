from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import FeatureBase

"""
itemの直近xx日間のinter, click, cart, orderの数のランキング
- aid
- item_inter_rank_3d
- item_click_rank_3d
- item_cart_rank_3d
- item_order_rank_3d
- item_inter_rank_7d
- item_click_rank_7d
- item_cart_rank_7d
- item_order_rank_7d
- item_inter_rank_14d
- item_click_rank_14d
- item_cart_rank_14d
- item_order_rank_14d
- item_inter_rank_21d
- item_click_rank_21d
- item_cart_rank_21d
- item_order_rank_21d
"""


class ItemInterRank(FeatureBase):
    def __init__(self, root_dir: Path, data_loader: OttoDataLoader):
        super().__init__(
            root_dir=root_dir,
            data_loader=data_loader,
            key_cols=["aid"],
            dtype=pl.Int32,
            fillna=0,
        )

    def _fit(self) -> pl.DataFrame:
        train_df = self.data_loader.get_train_df()

        feat_dfs = []
        for last_days in tqdm([3, 7, 14, 21]):
            for t, event_type in self.event_types:
                feat_df = (
                    train_df.filter(
                        pl.col("ts") >= pl.max("ts") - last_days * 24 * 60 * 60
                    )
                    .filter(pl.col("type").is_in(t))
                    .groupby("aid")
                    .agg(pl.count("type").alias("item_count"))
                    .sort("item_count", reverse=True)
                )

                feat_df = feat_df.with_column(
                    pl.Series(np.arange(1, len(feat_df) + 1))
                    .cast(pl.Int32)
                    .alias(f"item_{event_type}_rank_{last_days}d")
                )[["aid", f"item_{event_type}_rank_{last_days}d"]]

                feat_dfs.append(feat_df)

        feat_df = feat_dfs[0]
        for df in feat_dfs[1:]:
            feat_df = feat_df.join(df, on="aid", how="outer")
        feat_df = feat_df.fill_null(0)

        return feat_df
