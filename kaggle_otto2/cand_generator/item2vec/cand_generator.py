import gc
from pathlib import Path

import polars as pl

from kaggle_otto2.cand_generator.base.cand_generator_base import \
    CandGeneratorBase
from kaggle_otto2.cand_generator.item2vec.enums import SeedType
from kaggle_otto2.cand_generator.item2vec.trainer import Item2VecTrainer
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.util import TimeUtil


class Item2VecCandGenerator(CandGeneratorBase):
    def __init__(
        self,
        root_dir: Path,
        config: Config,
        data_loader: OttoDataLoader,
        seed_type: SeedType,
    ):
        self.short_cg_name = "item2vec"
        self.seed_type = seed_type
        cg_name = f"{self.short_cg_name}_{self.seed_type}"
        super().__init__(
            root_dir=root_dir,
            data_loader=data_loader,
            config=config,
            cg_name=cg_name,
            cg_weight=config.yaml.cg[cg_name].get("cg_weight", 1.0),
            dtype=pl.Float32,
        )
        self.trainer = Item2VecTrainer(data_loader=data_loader, config=config)
        self.model_dir = root_dir / "cand_generator" / self.short_cg_name
        self.gen_cand_topk = config.yaml.cg[cg_name].get("gen_cand_topk", 20)

    def fit(self):
        self.trainer.fit(self.model_dir)

    def gen_cand_df(self):
        with TimeUtil.timer("predict"):
            cand_df = self.trainer.predict(
                self.model_dir, topk=self.cand_topk, seed_type=self.seed_type
            ).with_columns(
                pl.col("score").cast(pl.Float32).alias(f"{self.cg_name}_score")
            )

        with TimeUtil.timer("save cand_df"):
            cand_df = self.postprocess_cand_df(cand_df)
            cand_df.write_parquet(self.data_dir / "cand_df.parquet")
            del cand_df
            gc.collect()
