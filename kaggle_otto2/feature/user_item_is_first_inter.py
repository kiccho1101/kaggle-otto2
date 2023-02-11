from pathlib import Path

import polars as pl

from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import FeatureBase

"""
userの最初/最後のイベントかどうか
- session
- aid
- user_item_is_first_inter
- user_item_is_last_inter
- user_item_is_first_click
- user_item_is_last_click
- user_item_is_first_cart
- user_item_is_last_cart
- user_item_is_first_order
- user_item_is_last_order
"""


class UserItemIsFirstInter(FeatureBase):
    def __init__(self, root_dir: Path, data_loader: OttoDataLoader):
        super().__init__(
            root_dir=root_dir,
            key_cols=["session", "aid"],
            data_loader=data_loader,
            fillna=0,
            dtype=pl.Int8,
        )

    def _fit(self):
        feat_dfs = []

        test_df = self.data_loader.get_test_df().to_pandas()
        test_df = test_df.sort_values(
            ["session", "ts"], ascending=[True, False]
        ).reset_index(drop=True)
        test_df["value"] = 1
        test_df["value"] = test_df["value"].astype("int8")

        for t, event_name in self.event_types:
            feat_dfs.append(
                pl.DataFrame(
                    test_df[test_df["type"].isin(t)]
                    .groupby("session")[["session", "aid", "value"]]
                    .head(1)
                    .rename(columns={"value": f"user_item_is_first_{event_name}"})
                )
            )
            feat_dfs.append(
                pl.DataFrame(
                    test_df[test_df["type"].isin(t)]
                    .groupby("session")[["session", "aid", "value"]]
                    .tail(1)
                    .rename(columns={"value": f"user_item_is_last_{event_name}"})
                )
            )

        feat_df = feat_dfs[0]
        for df in feat_dfs[1:]:
            feat_df = feat_df.join(df, on=["session", "aid"], how="left")

        feat_df = feat_df.with_columns(
            [
                pl.col(col).cast(pl.Int8)
                for col in feat_df.columns
                if col != "session" and col != "aid"
            ]
        ).fill_null(0)

        return feat_df
