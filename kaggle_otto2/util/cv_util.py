import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold


class CvUtil:
    @staticmethod
    def split_kfold(
        cand_df: pl.DataFrame, n_splits: int, target_col: str = "target_click"
    ) -> pl.DataFrame:
        cand_df = cand_df.with_column(pl.lit(0).cast(pl.Int8).alias("fold"))
        kf = GroupKFold(n_splits=n_splits)
        folds = np.zeros(len(cand_df))
        for fold, (train_idx, valid_idx) in enumerate(
            kf.split(cand_df, cand_df[target_col], groups=cand_df["session"])
        ):
            folds[valid_idx] = fold
        cand_df = cand_df.with_column(pl.Series("fold", folds).cast(pl.Int8))
        return cand_df
