from pathlib import Path

import pandas as pd
import polars as pl

from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import FeatureBase

"""
userの最初/最後のイベントの曜日
- user_first_inter_dow
- user_first_click_dow
- user_first_cart_dow
- user_first_order_dow
- user_last_inter_dow
- user_last_click_dow
- user_last_cart_dow
- user_last_order_dow
"""


class UserInterDow(FeatureBase):
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
        test_df["ts_dow"] = test_df["ts_dt"].dt.day_of_week

        test_df = test_df.sort_values(
            ["session", "ts"], ascending=[True, False]
        ).reset_index(drop=True)

        feat_dfs = []
        for t, event_name in self.event_types:
            feat_dfs.append(
                pl.DataFrame(
                    test_df[test_df["type"].isin(t)]
                    .groupby("session")[["session", "ts_dow"]]
                    .head(1)
                    .rename(columns={"ts_dow": f"user_first_{event_name}_dow"})
                )
            )
            feat_dfs.append(
                pl.DataFrame(
                    test_df[test_df["type"].isin(t)]
                    .groupby("session")[["session", "ts_dow"]]
                    .tail(1)
                    .rename(columns={"ts_dow": f"user_last_{event_name}_dow"})
                )
            )

        feat_df = feat_dfs[0]
        for df in feat_dfs[1:]:
            feat_df = feat_df.join(df, on="session", how="left")
        feat_df = feat_df.with_columns(
            [pl.col(col).cast(pl.Int8) for col in feat_df.columns if col != "session"]
        )
        return feat_df
