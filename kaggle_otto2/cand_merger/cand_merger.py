import gc
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
from tqdm import tqdm

from kaggle_otto2.cand_generator.base.cand_generator_base import CandGeneratorBase
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.util import CvUtil, GlobalUtil, TimeUtil


class CandMerger(CandGeneratorBase):
    def __init__(self, root_dir: Path, config: Config, data_loader: OttoDataLoader):
        super().__init__(
            root_dir=root_dir,
            data_loader=data_loader,
            config=config,
            cg_name="cand_merger",
        )
        self.config = config

    def merge(self, cand_generators: List[CandGeneratorBase]):
        for cg in cand_generators:
            print(cg.cg_name)
            print(cg.scan().head().collect())

        with TimeUtil.timer("get session aid list"):
            cg_chunks = np.array_split(cand_generators, 4)
            cand_df = pl.DataFrame()
            with tqdm(cg_chunks, dynamic_ncols=True) as tepoch:
                for cg_chunk in tepoch:
                    cand_df = pl.concat(
                        [
                            cand_df,
                            pl.concat(
                                [
                                    cg.scan()
                                    .groupby("session")
                                    .head(cg.gen_cand_topk)
                                    .select(
                                        [
                                            pl.col("session"),
                                            pl.col("aid"),
                                        ]
                                    )
                                    for cg in cg_chunk
                                ]
                            )
                            .unique()
                            .collect(),
                        ]
                    ).unique()
                    _, m, p = GlobalUtil.get_metric()
                    tepoch.set_postfix(mem=f"{m:.1f}GB({p:.1f}%)")

        if self.config.is_cv:
            with TimeUtil.timer("add target cols"):
                for t, target_col in tqdm(
                    [(0, "target_click"), (1, "target_cart"), (2, "target_order")]
                ):
                    test_labels = self.data_loader.get_test_labels()
                    target_df = (
                        test_labels.filter(pl.col("type") == t)[
                            ["session", "ground_truth"]
                        ]
                        .explode(["ground_truth"])
                        .select(
                            [
                                pl.col("session").cast(pl.Int32),
                                pl.col("ground_truth").cast(pl.Int32).alias("aid"),
                                pl.lit(1).alias(target_col),
                            ]
                        )
                    )
                    cand_df = cand_df.join(
                        target_df, on=["session", "aid"], how="left"
                    ).with_column(pl.col(target_col).fill_null(0).cast(pl.Int8))
                    del target_df
                    gc.collect()

            with TimeUtil.timer("add no_target cols"):
                for target_col in tqdm(["target_click", "target_cart", "target_order"]):
                    cand_df = cand_df.with_column(
                        (pl.sum(target_col).over("session") == 0)
                        .cast(pl.Int8)
                        .alias(f"no_{target_col}")
                    )

            with TimeUtil.timer("drop no target sessions"):
                print("before dropping:", cand_df.shape)
                cand_df = cand_df.filter(
                    (pl.col("no_target_click") == 0)
                    | (pl.col("no_target_cart") == 0)
                    | (pl.col("no_target_order") == 0)
                )
                print("after dropping:", cand_df.shape)

            with TimeUtil.timer("split kfold"):
                cand_df = CvUtil.split_kfold(cand_df, 3, "target_click")

        with TimeUtil.timer("join selected"):
            with tqdm(cand_generators, dynamic_ncols=True) as tepoch:
                for cg in tepoch:
                    cand_df = cand_df.join(
                        cg.scan()
                        .groupby("session")
                        .head(cg.gen_cand_topk)
                        .select(
                            [
                                pl.col("session"),
                                pl.col("aid"),
                                pl.lit(1)
                                .cast(pl.Int8)
                                .alias(f"{cg.cg_name}_score_selected"),
                            ]
                        )
                        .collect(),
                        on=["session", "aid"],
                        how="left",
                    ).with_column(pl.col(f"{cg.cg_name}_score_selected").fill_null(0))
                    _, m, p = GlobalUtil.get_metric()
                    tepoch.set_postfix(mem=f"{m:.1f}GB({p:.1f}%)")

        with TimeUtil.timer("join scores"):
            with tqdm(cand_generators, dynamic_ncols=True) as tepoch:
                for cg in tepoch:
                    _cand_df = cg.scan().select(
                        [
                            pl.col("session"),
                            pl.col("aid"),
                            pl.col(f"{cg.cg_name}_score"),
                            pl.col(f"{cg.cg_name}_score_rank"),
                            pl.col(f"{cg.cg_name}_score_min_rank"),
                        ]
                    )
                    cand_df = cand_df.join(
                        _cand_df.collect(),
                        on=["session", "aid"],
                        how="left",
                    ).with_columns(
                        [
                            pl.col(f"{cg.cg_name}_score").fill_null(-1).cast(cg.dtype),
                            pl.col(f"{cg.cg_name}_score_rank")
                            .fill_null(-1)
                            .cast(pl.Int16),
                            pl.col(f"{cg.cg_name}_score_min_rank")
                            .fill_null(-1)
                            .cast(pl.Int16),
                        ]
                    )
                    _, m, p = GlobalUtil.get_metric()
                    tepoch.set_postfix(mem=f"{m:.1f}GB({p:.1f}%)")

        with TimeUtil.timer("add cand_feature_selected_num"):
            cand_df = cand_df.with_columns(
                [
                    # 何個の候補生成手法で選ばれたか
                    pl.sum(
                        [
                            pl.col(f"{cg.cg_name}_score_selected")
                            for cg in cand_generators
                        ]
                    )
                    .cast(pl.Int8)
                    .alias("cand_feature_selected_num"),
                    # 何個の候補生成手法で選ばれたか(重み付けあり)
                    pl.sum(
                        [
                            pl.col(f"{cg.cg_name}_score_selected") * cg.cg_weight
                            for cg in cand_generators
                        ]
                    )
                    .cast(pl.Float32)
                    .alias("cand_feature_selected_score"),
                ]
            )

        with TimeUtil.timer("add cand_feature_score"):
            cand_df = cand_df.with_columns(
                [
                    pl.col("cand_feature_selected_score")
                    .rank(method="ordinal", reverse=True)
                    .over("session")
                    .cast(pl.Int16)
                    .alias("cand_feature_selected_rank"),
                    pl.col("cand_feature_selected_score")
                    .rank(method="min", reverse=True)
                    .over("session")
                    .cast(pl.Int16)
                    .alias("cand_feature_selected_min_rank"),
                ]
            )

        with TimeUtil.timer("save cand_df"):
            cand_df.write_parquet(self.data_dir / "cand_df.parquet")
            del cand_df
            gc.collect()
