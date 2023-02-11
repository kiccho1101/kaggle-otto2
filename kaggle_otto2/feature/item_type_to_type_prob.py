from pathlib import Path

import polars as pl
from tqdm import tqdm

from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import FeatureBase

"""
itemの直近xx日間のclick→cart/order, cart→orderの確率
- aid
- item_click_to_order_prob_3d
- item_click_to_order_prob_7d
- item_click_to_order_prob_14d
- item_click_to_order_prob_21d
- item_click_to_cart_prob_3d
- item_click_to_cart_prob_7d
- item_click_to_cart_prob_14d
- item_click_to_cart_prob_21d
- item_cart_to_order_prob_3d
- item_cart_to_order_prob_7d
- item_cart_to_order_prob_14d
- item_cart_to_order_prob_21d
"""


class ItemTypeToTypeProb(FeatureBase):
    def __init__(self, root_dir: Path, data_loader: OttoDataLoader):
        super().__init__(
            root_dir=root_dir,
            data_loader=data_loader,
            key_cols=["aid"],
            dtype=pl.Float32,
            fillna=-1.0,
        )

    def _fit(self) -> pl.DataFrame:
        train_df = self.data_loader.get_train_df()

        feat_dfs = []
        for last_days in tqdm([3, 7, 14, 21]):
            feat_df = (
                train_df.filter(
                    pl.col("ts") < pl.max("ts") - last_days * 24 * 60 * 60
                )
                .groupby(["aid"])
                .pivot(pivot_column="type", values_column=["aid"])
                .count()
                .fill_null(0)
                .with_columns(
                    [
                        pl.when(pl.col("0") == 0)
                        .then(-1.0)
                        .otherwise(pl.col("1") / pl.col("0"))
                        .cast(pl.Float32)
                        .alias(f"item_click_to_cart_prob_{last_days}d"),
                        pl.when(pl.col("0") == 0)
                        .then(-1.0)
                        .otherwise(pl.col("2") / pl.col("0"))
                        .cast(pl.Float32)
                        .alias(f"item_click_to_order_prob_{last_days}d"),
                        pl.when(pl.col("1") == 0)
                        .then(-1.0)
                        .otherwise(pl.col("2") / pl.col("1"))
                        .cast(pl.Float32)
                        .alias(f"item_cart_to_order_prob_{last_days}d"),
                    ]
                )[
                    [
                        "aid",
                        f"item_click_to_cart_prob_{last_days}d",
                        f"item_click_to_order_prob_{last_days}d",
                        f"item_cart_to_order_prob_{last_days}d",
                    ]
                ]
            )
            feat_dfs.append(feat_df)

        feat_df = feat_dfs[0]
        for df in feat_dfs[1:]:
            feat_df = feat_df.join(df, on="aid", how="outer")
        feat_df = feat_df.fill_null(-1.0).with_column(pl.col("aid").cast(pl.Int32))

        return feat_df
