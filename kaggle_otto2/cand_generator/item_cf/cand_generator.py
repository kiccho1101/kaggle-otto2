import gc
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from tqdm import tqdm

from kaggle_otto2.cand_generator import CandGeneratorBase
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.util import TimeUtil

AggType = Literal["sum", "min", "mean", "max", "std", "last"]


class ItemCFCandGenerator(CandGeneratorBase):
    def __init__(
        self,
        root_dir: Path,
        data_loader: OttoDataLoader,
        config: Config,
        agg_method: AggType,
        source_types=[0, 1, 2],
        target_types=[0, 1, 2],
        use_aid_pop=True,
        use_ts_diff=True,
        use_trend=True,
        use_iif=True,
        use_future=True,
    ):
        self.source_types = source_types
        self.target_types = target_types
        self.agg_method = agg_method
        self.use_aid_pop = use_aid_pop
        self.use_ts_diff = use_ts_diff
        self.use_trend = use_trend
        self.use_iif = use_iif
        self.use_future = use_future
        self.short_cg_name = "item_cf_s{}_t{}_ap{}_td{}_tr{}_iif{}_f{}".format(
            "".join([str(x) for x in source_types]),
            "".join([str(x) for x in target_types]),
            self.use_aid_pop,
            self.use_ts_diff,
            self.use_trend,
            self.use_iif,
            self.use_future,
        )
        cg_name = "{}_{}".format(
            self.short_cg_name,
            self.agg_method,
        )
        super().__init__(
            root_dir=root_dir,
            data_loader=data_loader,
            config=config,
            cg_name=cg_name,
            cg_weight=0.7,
            dtype=pl.Float32,
        )
        self.gen_cand_topk = 40
        self.inference_split_num = 3
        self.pair_df_dir = self.root_dir / "cand_generator" / self.short_cg_name
        self.pair_df_dir.mkdir(parents=True, exist_ok=True)

    def fit(self):
        with TimeUtil.timer("calc pair_df"):
            with TimeUtil.timer("load train_df"):
                train_df = self.data_loader.get_train_df()
                train_df = train_df.with_column(
                    pl.count("aid").over("session").alias("session_aid_count")
                )

            with TimeUtil.timer("drop duplicates"):
                train_df = train_df.sort(["session", "ts"], reverse=True)
                train_df = train_df.unique(subset=["session", "aid", "type"])
                aid_counts = train_df["aid"].value_counts()
                gc.collect()

            with TimeUtil.timer("join"):
                train_df = (
                    train_df.join(train_df, on="session")
                    .drop("session")
                    .drop("session_aid_count_right")
                )
                gc.collect()

            if self.use_future:
                with TimeUtil.timer("filter by ts future"):
                    train_df = train_df.filter(pl.col("ts") < pl.col("ts_right"))

            with TimeUtil.timer("filter by types"):
                train_df = (
                    train_df[: len(train_df) - 1]
                    .filter(
                        pl.col("type").is_in(self.source_types)
                        & pl.col("type_right").is_in(self.target_types)
                    )
                    .drop("type")
                    .drop("type_right")
                )
                gc.collect()

            with TimeUtil.timer("add aid_count"):
                train_df = train_df.join(aid_counts, on="aid", how="left")
                train_df = train_df.join(
                    aid_counts, left_on="aid_right", right_on="aid", how="left"
                )
                gc.collect()

            with TimeUtil.timer("add item_count_coef column"):
                train_df = (
                    train_df.with_column(
                        (
                            1.0
                            / (pl.col("counts") * pl.col("counts_right") + 2)
                            .sqrt()
                            .log()
                            if self.use_aid_pop
                            else pl.lit(1.0)
                        )
                        .alias("item_count_coef")
                        .cast(pl.Float32)
                    )
                    .drop("counts")
                    .drop("counts_right")
                )
                gc.collect()

            with TimeUtil.timer("add trend_coef column"):
                train_df = train_df.with_column(
                    (
                        1.0 / (pl.max("ts") - pl.col("ts_right") + 1)
                        if self.use_trend
                        else pl.lit(1.0)
                    )
                    .alias("trend_coef")
                    .cast(pl.Float32)
                )
                gc.collect()

            with TimeUtil.timer("add ts_diff_coef column"):
                train_df = (
                    train_df.with_column(
                        (
                            1
                            / 2
                            ** (((pl.col("ts") - pl.col("ts_right")) / 60 / 60).abs())
                        )
                        .alias("ts_diff_coef")
                        .cast(pl.Float32)
                    )
                    .drop("ts")
                    .drop("ts_right")
                )
                gc.collect()

            with TimeUtil.timer("add iif column"):
                train_df = train_df.with_column(
                    (
                        1 / (1 + pl.col("session_aid_count")).log()
                        if self.use_iif
                        else pl.lit(1.0)
                    )
                    .alias("iif_coef")
                    .cast(pl.Float32)
                ).drop("session_aid_count")

            with TimeUtil.timer("add weight column"):
                train_df = train_df.with_column(
                    (pl.col("item_count_coef") * pl.col("trend_coef")).alias("weight")
                ).drop(["item_count_coef", "trend_coef"])
                train_df = train_df.with_column(
                    (pl.col("weight") * pl.col("ts_diff_coef"))
                    .cast(pl.Float32)
                    .alias("weight")
                ).drop("ts_diff_coef")
                train_df = train_df.with_column(
                    (pl.col("weight") * pl.col("iif_coef"))
                    .cast(pl.Float32)
                    .alias("weight")
                ).drop("iif_coef")

            with TimeUtil.timer("save pair_df"):
                train_df.write_parquet(self.pair_df_dir / "pair_df.parquet")
                del train_df
                gc.collect()

            with TimeUtil.timer("read pair_df"):
                pair_df = pl.read_parquet(
                    self.pair_df_dir / "pair_df.parquet"
                ).with_columns(
                    [
                        pl.col("aid").cast(pl.Int32),
                        pl.col("aid_right").cast(pl.Int32),
                    ]
                )

            with TimeUtil.timer("calc sum of weight"):
                pair_df = pair_df.groupby(["aid", "aid_right"]).agg(pl.sum("weight"))

            with TimeUtil.timer("sort"):
                pair_df = pair_df.sort(["aid", "weight"], reverse=True)

            with TimeUtil.timer("drop rows >= 300"):
                print("before drop rows:", pair_df.shape)
                pair_df = pair_df.groupby("aid").head(300)
                print("after drop rows:", pair_df.shape)

            with TimeUtil.timer("save pair_df"):
                pair_df.write_parquet(self.pair_df_dir / "pair_df.parquet")
                del pair_df
                gc.collect()

    def gen_cand_df(self):
        with TimeUtil.timer("load"):
            pair_df = pl.read_parquet(self.pair_df_dir / "pair_df.parquet")
            test_df = self.data_loader.get_test_df()

        test_df = test_df.filter(pl.col("type").is_in(self.source_types)).with_columns(
            [
                pl.col("session").cast(pl.Int32),
                pl.col("aid").cast(pl.Int32),
            ]
        )
        sessions = test_df["session"].unique()
        session_chunks = np.array_split(sessions, self.inference_split_num)

        if self.agg_method == "last":
            test_df = (
                test_df.sort(["session", "ts"], reverse=True).groupby("session").head(1)
            )

        cand_df = pl.DataFrame()
        with TimeUtil.timer("get test_aids"):
            for session_chunk in tqdm(session_chunks):
                test_aids = test_df.filter(
                    pl.col("session").is_in(session_chunk.tolist())
                ).join(pair_df[["aid", "aid_right", "weight"]], on="aid", how="left")
                if self.agg_method == "mean":
                    test_aids = test_aids.groupby(["session", "aid_right"]).agg(
                        pl.mean("weight")
                    )
                elif self.agg_method == "max":
                    test_aids = test_aids.groupby(["session", "aid_right"]).agg(
                        pl.max("weight")
                    )
                elif self.agg_method == "min":
                    test_aids = test_aids.groupby(["session", "aid_right"]).agg(
                        pl.min("weight")
                    )
                elif self.agg_method == "std":
                    test_aids = test_aids.groupby(["session", "aid_right"]).agg(
                        pl.std("weight")
                    )
                elif self.agg_method == "count":
                    test_aids = test_aids.groupby(["session", "aid_right"]).agg(
                        pl.count("weight")
                    )
                else:
                    test_aids = test_aids.groupby(["session", "aid_right"]).agg(
                        pl.sum("weight")
                    )
                test_aids = test_aids.sort(["session", "weight"], reverse=True)
                _cand_df = (
                    test_aids.filter(pl.col("aid_right").is_not_null())
                    .groupby("session")
                    .head(self.cand_topk)
                    .rename(
                        {
                            "aid_right": "aid",
                            "weight": f"{self.cg_name}_score",
                        }
                    )
                    .with_column(pl.col(f"{self.cg_name}_score").cast(pl.Float32))
                )
                cand_df = pl.concat([cand_df, _cand_df])

        with TimeUtil.timer("save cand_df"):
            cand_df = self.postprocess_cand_df(cand_df)
            cand_df.write_parquet(self.data_dir / "cand_df.parquet")
            del cand_df
            gc.collect()
