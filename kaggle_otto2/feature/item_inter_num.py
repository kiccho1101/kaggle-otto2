from pathlib import Path

import polars as pl
from tqdm import tqdm

from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import FeatureBase

"""
itemの直近xx日間のinter, click, cart, orderの数
- aid
- item_inter_num_3d
- item_click_num_3d
- item_cart_num_3d
- item_order_num_3d
- item_inter_num_7d
- item_click_num_7d
- item_cart_num_7d
- item_order_num_7d
- item_inter_num_14d
- item_click_num_14d
- item_cart_num_14d
- item_order_num_14d
- item_inter_num_21d
- item_click_num_21d
- item_cart_num_21d
- item_order_num_21d
- item_buy_num_3d
- item_buy_num_7d
- item_buy_num_14d
- item_buy_num_21d
"""


class ItemInterNum(FeatureBase):
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
                    .agg(
                        pl.count("type")
                        .alias(f"item_{event_type}_num_{last_days}d")
                        .cast(pl.Int32)
                    )
                )
                feat_dfs.append(feat_df)

        feat_df = feat_dfs[0]
        for df in feat_dfs[1:]:
            feat_df = feat_df.join(df, on="aid", how="outer")
        feat_df = feat_df.fill_null(0)

        return feat_df
