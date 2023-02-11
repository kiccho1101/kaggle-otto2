import gc
from pathlib import Path

import polars as pl

from kaggle_otto2.cand_generator.base.cand_generator_base import \
    CandGeneratorBase
from kaggle_otto2.cand_generator.user_mf.trainer import UserMFTrainer
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.util import FileUtil, TimeUtil


class UserMFCandGenerator(CandGeneratorBase):
    def __init__(self, root_dir: Path, config: Config, data_loader: OttoDataLoader):
        cg_name = "user_mf"
        super().__init__(
            root_dir=root_dir,
            data_loader=data_loader,
            config=config,
            cg_name=cg_name,
            cg_weight=config.yaml.cg[cg_name].get("cg_weight", 3.0),
            dtype=pl.Float32,
        )
        self.trainer = UserMFTrainer(data_loader=data_loader, config=config)
        self.model_dir = self.data_dir
        self.gen_cand_topk = config.yaml.cg[cg_name].get("gen_cand_topk", 40)

    def fit(self):
        self.trainer.fit(self.model_dir)

    def gen_cand_df(self):
        with TimeUtil.timer("load model"):
            self.trainer.load_model(self.model_dir)

        with TimeUtil.timer("predict"):
            idx2aid = FileUtil.load_pickle(self.data_dir / "idx2aid.pkl")
            session2idx = FileUtil.load_pickle(self.data_dir / "session2idx.pkl")
            cand_df = self.trainer.model.predict(
                self.data_loader, idx2aid, session2idx, topk=self.cand_topk
            ).with_columns(
                pl.col("score").cast(pl.Float32).alias(f"{self.cg_name}_score")
            )

        with TimeUtil.timer("save cand_df"):
            cand_df = self.postprocess_cand_df(cand_df)
            cand_df.write_parquet(self.data_dir / "cand_df.parquet")
            del cand_df
            gc.collect()
