from pathlib import Path
from typing import Any, Dict, Optional, Union

import polars as pl

from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.util import EvaluateUtil, TimeUtil


class CandGeneratorBase:
    def __init__(
        self,
        root_dir: Union[str, Path],
        cg_name: str,
        config: Config,
        data_loader: OttoDataLoader,
        cg_weight=1.0,
        dtype: Any = pl.Float32,
    ):
        self.root_dir = Path(root_dir)
        self.cg_name = cg_name
        self.config = config
        self.data_loader = data_loader
        self.cg_weight = cg_weight
        self.data_dir = self.root_dir / "cand_generator" / self.cg_name
        self.data_dir.mkdir(exist_ok=True, parents=True)

        self.cand_topk = 300
        self.gen_cand_topk = 20

        self.dtype = dtype

    def load(self) -> pl.DataFrame:
        return pl.read_parquet(self.data_dir / "cand_df.parquet")

    def scan(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.data_dir / "cand_df.parquet")

    def postprocess_cand_df(self, cand_df: pl.DataFrame) -> pl.DataFrame:
        score_col = f"{self.cg_name}_score"
        cand_df = cand_df.sort(["session", score_col], reverse=True).with_columns(
            [
                pl.col("session").cast(pl.Int32),
                pl.col("aid").cast(pl.Int32),
                pl.col(score_col).cast(self.dtype),
                # Score Rank (distinct)
                pl.col(score_col)
                .rank(method="ordinal", reverse=True)
                .over("session")
                .cast(pl.Int16)
                .alias(f"{score_col}_rank"),
                # Score Rank (min)
                pl.col(score_col)
                .rank(method="min", reverse=True)
                .over("session")
                .cast(pl.Int16)
                .alias(f"{score_col}_min_rank"),
            ]
        )
        return cand_df

    def calc_score(
        self, cand_df: Optional[pl.DataFrame] = None, topks=[20, 50, 100, 200, 300]
    ) -> Dict[str, Any]:
        recall_scores: Dict[str, Any] = {"cg_name": self.cg_name}
        with TimeUtil.timer(f"calc score [{self.cg_name}]"):
            test_labels = self.data_loader.get_test_labels()

            sorted_col = f"{self.cg_name}_score"
            sort_ascending = True
            if self.cg_name == "cand_merger":
                sorted_col = "cand_feature_selected_rank"
                sort_ascending = False

            if cand_df is None:
                cand_df = self.load()

            cand_df = (
                cand_df.sort(pl.col(sorted_col), reverse=sort_ascending)
                .groupby("session")
                .agg(pl.col("aid").alias("y_pred"))
            )
            pred_dfs = {
                "target_click": cand_df,
                "target_cart": cand_df,
                "target_order": cand_df,
            }

            for topk in topks:
                score, scores, null_ratio = EvaluateUtil.calc_score(
                    test_labels, pred_dfs, topk=topk
                )
                if topk == 20:
                    recall_scores[f"recall_click@{topk}"] = scores[0]
                    recall_scores[f"recall_cart@{topk}"] = scores[1]
                    recall_scores[f"recall_order@{topk}"] = scores[2]
                recall_scores[f"recall@{topk}"] = score
                recall_scores["null_ratio"] = null_ratio

            pl.Config.set_tbl_cols(10)
            print(pl.DataFrame([recall_scores]))

        return recall_scores
