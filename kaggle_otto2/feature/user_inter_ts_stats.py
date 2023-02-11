from pathlib import Path

import polars as pl

from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import FeatureBase


# ユーザごとのsession tsの統計量
# cv, lbでshift しないようにstanderizeをかませる
# groupby 内で window関数は使えないのでses, aidで計算→unique
def standerize(expr: pl.Expr, col: str) -> pl.Expr:
    return (expr - pl.mean(col)) / pl.std(col)


class UserInterTsStats(FeatureBase):
    def __init__(self, root_dir: Path, data_loader: OttoDataLoader):
        super().__init__(
            root_dir=root_dir,
            data_loader=data_loader,
            key_cols=["session"],
            fillna=0,
            dtype=pl.Float32,
        )

    def _fit(self):
        test_df = self.data_loader.get_test_df()
        feat_df = test_df.select(
            [
                "session",
                standerize(pl.col("ts").mean().over("session"), "ts")
                .alias("user_inter_ts_mean")
                .cast(pl.Float32),
                standerize(pl.col("ts").min().over("session"), "ts")
                .alias("user_inter_ts_min")
                .cast(pl.Float32),
                standerize(pl.col("ts").max().over("session"), "ts")
                .alias("user_inter_ts_max")
                .cast(pl.Float32),
                standerize(pl.col("ts").std().over("session"), "ts")
                .alias("user_inter_ts_std")
                .cast(pl.Float32),
            ]
        ).unique()
        return feat_df
