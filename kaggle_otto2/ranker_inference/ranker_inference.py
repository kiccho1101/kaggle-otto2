import gc
from pathlib import Path
from typing import List

import lightgbm as lgb
import pandas as pd
import polars as pl
import xgboost as xgb
from catboost import CatBoost
from tqdm import tqdm

from kaggle_otto2.cand_merger import NegSampler
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import FeatureMerger
from kaggle_otto2.ranker_trainer.enums import ModelType
from kaggle_otto2.util import TimeUtil


class RankerInference:
    def __init__(
        self,
        output_dir: Path,
        data_loader: OttoDataLoader,
        feature_merger: FeatureMerger,
        neg_sampler: NegSampler,
        config: Config,
        model_types: List[ModelType] = [
            "lgbm",
            "catboost",
            "catboost_ranker",
            "xgb",
        ],
        targets=[
            "target_inter",
            "target_buy",
            "target_click",
            "target_cart",
            "target_order",
        ],
    ):
        self.output_dir = output_dir
        self.data_loader = data_loader
        self.feature_merger = feature_merger
        self.neg_sampler = neg_sampler
        self.config = config
        self.model_types = model_types
        self.targets = targets

    def inference(self, features: List[str]):
        n_splits = 3
        for fold in range(n_splits):
            with TimeUtil.timer(f"Inference Fold: {fold}/{n_splits-1}"):
                self.inference_fold(fold, features)

    def inference_fold(self, fold: int, features: List[str]):
        cols = [pl.col("session"), pl.col("aid")] + [pl.col(f) for f in features]

        with TimeUtil.timer(f"load valid_df [{fold}]"):
            valid_df = self.get_valid_df(fold, cols)
            print("valid_df:", valid_df.shape)

        with TimeUtil.timer("add score cols"):
            for model_type in self.model_types:
                for target_col in self.targets:
                    if model_type == "xgb" and target_col in [
                        "target_inter",
                        "target_click",
                    ]:
                        continue
                    with TimeUtil.timer(f"inference [{model_type} {target_col}]"):
                        model = self.load_model(model_type, target_col, fold)
                        if model_type == "catboost":
                            BATCH_SIZE = 50_000_000
                            for bucket in tqdm(range(0, len(valid_df), BATCH_SIZE)):
                                valid_df.loc[
                                    bucket : bucket + BATCH_SIZE,
                                    f"{model_type}_{target_col}_pred_score",
                                ] = model.predict(
                                    valid_df.loc[
                                        bucket : bucket + BATCH_SIZE, features
                                    ],
                                    prediction_type="Probability",
                                ).T[
                                    1
                                ]
                        elif model_type == "xgb":
                            BATCH_SIZE = 50_000_000
                            for bucket in tqdm(range(0, len(valid_df), BATCH_SIZE)):
                                valid_df.loc[
                                    bucket : bucket + BATCH_SIZE,
                                    f"{model_type}_{target_col}_pred_score",
                                ] = model.predict(
                                    xgb.DMatrix(
                                        valid_df.loc[
                                            bucket : bucket + BATCH_SIZE, features
                                        ]
                                    )
                                )
                        else:
                            BATCH_SIZE = 100_000_000
                            for bucket in tqdm(range(0, len(valid_df), BATCH_SIZE)):
                                valid_df.loc[
                                    bucket : bucket + BATCH_SIZE,
                                    f"{model_type}_{target_col}_pred_score",
                                ] = model.predict(
                                    valid_df.loc[bucket : bucket + BATCH_SIZE, features]
                                )
                        valid_df[f"{model_type}_{target_col}_pred_score"] = valid_df[
                            f"{model_type}_{target_col}_pred_score"
                        ].astype("float32")

                    with TimeUtil.timer(f"save [{model_type} {target_col}]"):
                        valid_df[
                            [
                                "session",
                                "aid",
                                f"{model_type}_{target_col}_pred_score",
                            ]
                        ].reset_index(drop=True).to_parquet(
                            self.output_dir
                            / f"stack_df_{model_type}_{target_col}_{fold}.parquet"
                        )

    def load_model(self, model_type: str, target_col, fold: int):
        model_dir = self.config.dir_config.exp_output_dir
        if not self.config.is_cv:
            model_dir = self.config.dir_config.output_dir / "exp059_cv"
        if model_type == "lgbm":
            return lgb.Booster(
                model_file=model_dir / f"lgbm_{target_col}_fold{fold}.txt"
            )
        if model_type == "lgbm_cls":
            return lgb.Booster(
                model_file=model_dir / f"lgbm_cls_{target_col}_fold{fold}.txt"
            )
        if model_type == "catboost":
            model = CatBoost()
            model.load_model(model_dir / f"catboost_{target_col}_fold{fold}.txt")
            return model
        if model_type == "catboost_ranker":
            model = CatBoost()
            model.load_model(model_dir / f"catboost_ranker_{target_col}_fold{fold}.txt")
            return model
        if model_type == "xgb":
            model = xgb.Booster()
            model.load_model(model_dir / f"xgb_{target_col}_fold{fold}.txt")
            return model
        raise ValueError(f"Invalid model_type: {model_type}")

    def get_valid_df(self, fold, cols) -> pd.DataFrame:
        expr = self.feature_merger.scan()
        if self.config.is_cv:
            expr = expr.filter(pl.col("fold") == fold)
        return (
            expr.sort("session")
            .select(cols)
            .collect()
            .to_pandas()
            .reset_index(drop=True)
        )
