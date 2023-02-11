import gc
from pathlib import Path
from typing import List

import polars as pl
from tqdm import tqdm

from kaggle_otto2.cand_generator import CandGeneratorBase
from kaggle_otto2.cand_merger import CandMerger
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import FeatureBase
from kaggle_otto2.util import GlobalUtil, TimeUtil


class FeatureMerger(CandGeneratorBase):
    def __init__(self, root_dir: Path, data_loader: OttoDataLoader, config: Config):
        super().__init__(
            root_dir=root_dir,
            data_loader=data_loader,
            config=config,
            cg_name="feature_merger",
        )

    def merge(self, cand_merger: CandMerger, features: List[FeatureBase]):
        with TimeUtil.timer("merge"):
            with TimeUtil.timer("merge features"):
                feat_dfs = {}
                for feature in tqdm(features):
                    dict_key = ",".join(feature.key_cols)
                    feat_df = feature.load()
                    if dict_key not in feat_dfs:
                        feat_dfs[dict_key] = feature.load().with_columns(
                            [pl.col(key).cast(pl.Int32) for key in feature.key_cols]
                        )
                    else:
                        feat_dfs[dict_key] = feat_dfs[dict_key].join(
                            feature.load().with_columns(
                                [pl.col(key).cast(pl.Int32) for key in feature.key_cols]
                            ),
                            on=feature.key_cols,
                            how="outer",
                        )

            with TimeUtil.timer("load cand_merger cand_df"):
                cand_df = cand_merger.load()

            with TimeUtil.timer("merge features into cand_df"):
                keys = sorted(list(feat_dfs.keys()))
                print(keys)
                with tqdm(keys, dynamic_ncols=True) as tepoch:
                    for key in tepoch:
                        feat_df = feat_dfs[key]
                        cand_df = cand_df.join(feat_df, on=key.split(","), how="left")
                        _, m, p = GlobalUtil.get_metric()
                        tepoch.set_postfix(mem=f"{m:.1f}GB({p:.1f}%)")

            with TimeUtil.timer("fillna and convert dtypes"):
                for feature in tqdm(features):
                    cand_df = cand_df.with_columns(
                        [
                            pl.col(col).fill_null(feature.fillna).cast(feature.dtype)
                            for col in feature.load_columns()
                        ]
                    )

            with TimeUtil.timer("save cand_df"):
                cand_df.write_parquet(
                    self.data_dir / "cand_df.parquet",
                )

        del cand_df
        gc.collect()
