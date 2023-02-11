from pathlib import Path

import polars as pl

from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import FeatureBase

"""
itemがリピートしてclick/cart/orderされる確率
- aid
- item_multi_inter_prob
- item_multi_click_prob
- item_multi_cart_prob
- item_multi_order_prob
"""


class ItemMultiInterProb(FeatureBase):
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
        for t, event_type in self.event_types:
            feat_df = (
                train_df.filter(pl.col("type").is_in(t))
                .groupby(["session", "aid"])
                .count()
                .with_column(
                    pl.when(pl.col("count") > 1)
                    .then(1)
                    .otherwise(0)
                    .alias("is_duplicate")
                )
                .groupby("aid")
                .agg(
                    pl.mean("is_duplicate")
                    .cast(pl.Float32)
                    .alias(f"item_multi_{event_type}_prob")
                )
            )
            feat_dfs.append(feat_df)

        feat_df = feat_dfs[0]
        for df in feat_dfs[1:]:
            feat_df = feat_df.join(df, on="aid", how="outer")
        feat_df = feat_df.fill_null(-1.0).with_column(pl.col("aid").cast(pl.Int32))

        return feat_df
