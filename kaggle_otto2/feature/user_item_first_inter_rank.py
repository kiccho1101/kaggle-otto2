from pathlib import Path

import polars as pl

from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import FeatureBase

"""
userの最初/最後からのイベントの順番
- user_item_first_inter_rank
- user_item_first_click_rank
- user_item_first_cart_rank
- user_item_first_order_rank
- user_item_last_inter_rank
- user_item_last_click_rank
- user_item_last_cart_rank
- user_item_last_order_rank
"""


class UserItemFirstInterRank(FeatureBase):
    def __init__(self, root_dir: Path, data_loader: OttoDataLoader):
        super().__init__(
            root_dir=root_dir,
            key_cols=["session", "aid"],
            data_loader=data_loader,
            fillna=-1,
            dtype=pl.Int32,
        )

    def _fit(self):
        test_df = self.data_loader.get_test_df().to_pandas()

        feat_dfs = []

        test_df = test_df.sort_values(
            ["session", "ts"], ascending=[True, False]
        ).reset_index(drop=True)
        for t, event_name in self.event_types:
            test_df = test_df[test_df["type"].isin(t)]
            test_df["user_cumcount"] = test_df.groupby("session").cumcount()
            feat_dfs.append(
                pl.DataFrame(
                    test_df.drop_duplicates(["session", "aid"])[
                        ["session", "aid", "user_cumcount"]
                    ].rename(
                        columns={"user_cumcount": f"user_item_first_{event_name}_rank"}
                    )
                )
            )

        test_df = test_df.sort_values(
            ["session", "ts"], ascending=[True, True]
        ).reset_index(drop=True)

        for t, event_name in self.event_types:
            test_df = test_df[test_df["type"].isin(t)]
            test_df["user_cumcount"] = test_df.groupby("session").cumcount()
            feat_dfs.append(
                pl.DataFrame(
                    test_df.drop_duplicates(["session", "aid"])[
                        ["session", "aid", "user_cumcount"]
                    ].rename(
                        columns={"user_cumcount": f"user_item_last_{event_name}_rank"}
                    )
                )
            )

        feat_df = feat_dfs[0]
        for df in feat_dfs[1:]:
            feat_df = feat_df.join(df, on=["session", "aid"], how="left")

        feat_df = (
            feat_df.with_columns(
                [
                    pl.col(col).cast(pl.Int32)
                    for col in feat_df.columns
                    if col != "session" and col != "aid"
                ]
            )
            .fill_null(-1)
        )

        return feat_df
