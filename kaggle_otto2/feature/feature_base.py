from pathlib import Path
from typing import Any, List, Set

import polars as pl

from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.util import TimeUtil


class FeatureBase:
    def __init__(
        self,
        root_dir: Path,
        data_loader: OttoDataLoader,
        key_cols: List[str],
        fillna=-1,
        dtype: Any = pl.Float32,
    ):
        self.data_loader = data_loader
        self.fillna = fillna
        self.dtype = dtype
        self.key_cols = key_cols
        self.event_types = [
            ([0, 1, 2], "inter"),
            ([1, 2], "buy"),
            ([0], "click"),
            ([1], "cart"),
            ([2], "order"),
        ]
        self.name = self.__class__.__name__
        self.data_dir = root_dir / "feature_eng" / self.name
        self.data_dir.mkdir(exist_ok=True, parents=True)

    def _fit(self) -> pl.DataFrame:
        raise NotImplementedError()

    def fit(self):
        with TimeUtil.timer(self.name):
            feat_df = self._fit()
            self.save(feat_df)

    def save(self, feat_df: pl.DataFrame):
        feat_df.write_parquet(self.data_dir / "feat_df.parquet")

    def load(self) -> pl.DataFrame:
        return pl.read_parquet(self.data_dir / "feat_df.parquet")

    def load_columns(self) -> List[str]:
        return [
            col
            for col in pl.scan_parquet(self.data_dir / "feat_df.parquet").columns
            if col not in self.key_cols
        ]

    def merge(self, cand_df: pl.DataFrame, features: Set[str]) -> pl.DataFrame:
        feat_df = self.load()
        feature_cols = [f for f in feat_df.columns if f in features]
        if len(feature_cols) == 0:
            return cand_df
        cand_df = cand_df.join(
            feat_df[self.key_cols + feature_cols],
            on=self.key_cols,
            how="left",
        ).fill_null(self.fillna)
        return cand_df
