import gc
import itertools
import random
from pathlib import Path

import pandas as pd

from kaggle_otto2.cand_generator import CandGeneratorBase
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import FeatureMerger
from kaggle_otto2.util import TimeUtil


class NegSampler(CandGeneratorBase):
    def __init__(self, root_dir: Path, data_loader: OttoDataLoader, config: Config):
        super().__init__(
            root_dir=root_dir,
            data_loader=data_loader,
            config=config,
            cg_name="neg_sampler",
        )

    def neg_sampling(self, feature_merger: FeatureMerger):
        with TimeUtil.timer("neg sampling"):
            with TimeUtil.timer("load"):
                cand_df = feature_merger.load().to_pandas()

            with TimeUtil.timer("neg sample"):
                print("before negative sampling: ", cand_df.shape)
                pos_cand_df = cand_df.loc[
                    (cand_df["target_click"] == 1)
                    | (cand_df["target_order"] == 1)
                    | (cand_df["target_cart"] == 1)
                ]
                neg_cand_df = cand_df.loc[
                    (cand_df["target_click"] == 0)
                    & (cand_df["target_order"] == 0)
                    & (cand_df["target_cart"] == 0)
                ].sample(frac=0.5, random_state=77)
                cand_df = pd.concat(
                    [pos_cand_df, neg_cand_df], axis=0, ignore_index=True
                )

                # session内に0 ~ session sizeのランダムな番号を与えてsortすることでshuffleを高速に実現する
                cand_df = cand_df.sort_values(["session"]).reset_index(drop=True)
                busket_sizes = cand_df.groupby("session").size().values
                cand_df["order"] = list(
                    itertools.chain(
                        *[random.sample(list(range(v)), v) for v in busket_sizes]
                    )
                )
                cand_df = (
                    cand_df.sort_values(["session", "order"])
                    .reset_index(drop=True)
                    .drop(columns="order")
                )
                print("after negative sampling: ", cand_df.shape)
                del pos_cand_df, neg_cand_df
                gc.collect()

            with TimeUtil.timer("save cand_df"):
                cand_df.to_parquet(self.data_dir / "cand_df.parquet")
                del cand_df
                gc.collect()
