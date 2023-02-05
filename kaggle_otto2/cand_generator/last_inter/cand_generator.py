from pathlib import Path
from typing import Literal

import polars as pl

from kaggle_otto2.cand_generator import CandGeneratorBase
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.util import TimeUtil

InterType = Literal["inter", "buy", "click", "cart", "order"]


class LastInterCandGenerator(CandGeneratorBase):
    def __init__(
        self,
        root_dir: Path,
        config: Config,
        data_loader: OttoDataLoader,
        inter_type: InterType,
    ):
        super().__init__(
            root_dir=root_dir,
            data_loader=data_loader,
            config=config,
            cg_name=f"last_{inter_type}",
            cg_weight=2.0,
            dtype=pl.Int32,
        )
        self.inter_type = inter_type

    def gen_cand_df(self):
        with TimeUtil.timer(f"gen_cand_df [{self.inter_type}]"):
            type_map = {"click": 0, "cart": 1, "order": 2}

            if self.inter_type == "inter":
                cand_df = (
                    self.data_loader.get_test_df()
                    .groupby(["session", "aid"])
                    .agg(pl.count("ts").cast(pl.Int32).alias(f"{self.cg_name}_score"))
                )
            elif self.inter_type == "buy":
                cand_df = (
                    self.data_loader.get_test_df()
                    .filter(pl.col("type").is_in([1, 2]))
                    .groupby(["session", "aid"])
                    .agg(pl.count("ts").cast(pl.Int32).alias(f"{self.cg_name}_score"))
                )
            else:
                cand_df = (
                    self.data_loader.get_test_df()
                    .filter(pl.col("type") == type_map[self.inter_type])
                    .groupby(["session", "aid"])
                    .agg(pl.count("ts").cast(pl.Int32).alias(f"{self.cg_name}_score"))
                )

            assert list(cand_df.columns) == ["session", "aid", f"{self.cg_name}_score"]

            cand_df = self.postprocess_cand_df(cand_df)
            cand_df.write_parquet(self.data_dir / "cand_df.parquet")
