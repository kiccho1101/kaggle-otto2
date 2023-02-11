import gc
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from catboost import CatBoost, Pool

from kaggle_otto2.cand_merger import NegSampler
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import FeatureMerger
from kaggle_otto2.ranker.enums import RankerType
from kaggle_otto2.util import TimeUtil


class RankerTrainer:
    def __init__(
        self,
        output_dir: Path,
        data_loader: OttoDataLoader,
        feature_merger: FeatureMerger,
        neg_sampler: NegSampler,
        config: Config,
        model_type: RankerType,
    ):
        self.output_dir = output_dir
        self.data_loader = data_loader
        self.feature_merger = feature_merger
        self.neg_sampler = neg_sampler
        self.config = config
        self.model_type = model_type

    def fit(
        self,
        features: List[str],
        targets=[
            "target_inter",
            "target_buy",
            "target_click",
            "target_cart",
            "target_order",
        ],
    ) -> Dict[str, pd.DataFrame]:
        pred_score_dict = {}
        if self.model_type == "xgb":
            targets = [
                t for t in targets if t in ["target_buy", "target_cart", "target_order"]
            ]
        for target_col in targets:
            pred_score_dict[target_col] = self.fit_target(target_col, features)
        if self.config.is_dev:
            pred_dfs = self.get_pred_dfs(pred_score_dict)
            return pred_dfs
        return {}

    def fit_target(
        self,
        target_col: str,
        features: List[str],
    ) -> List[Tuple[int, int, float]]:
        pred_scores = []
        n_splits = 3
        for fold in range(n_splits):
            if not self.config.is_dev:
                if fold != 0:
                    continue
            with TimeUtil.timer(f"target_col: {target_col} Fold: {fold}/{n_splits-1}"):
                pred_scores.append(self.train_ranker(fold, features, target_col))
        if self.config.is_dev:
            pred_scores = np.concatenate(pred_scores)
        return pred_scores

    def train_ranker(
        self,
        fold: int,
        features: List[str],
        target_col: str,
    ) -> List[Tuple[int, int, float]]:
        cols = [
            pl.col("session"),
            pl.col("aid"),
            pl.col(target_col),
        ] + [pl.col(f) for f in features]

        with TimeUtil.timer(f"load train_df [{target_col} {fold}]"):
            train_df = self.get_train_df(target_col, fold, cols)

        with TimeUtil.timer(f"load neg_valid_df [{target_col} {fold}]"):
            neg_valid_df = self.get_neg_valid_df(target_col, fold, cols)

        with TimeUtil.timer(f"load valid_df [{target_col} {fold}]"):
            valid_df = self.get_valid_df(target_col, fold, cols)

        print("train_df.shape:", train_df.shape)
        print("neg_valid_df.shape:", neg_valid_df.shape)
        print("valid_df.shape:", valid_df.shape)

        if self.model_type == "lgbm":
            valid_pred = self.train_lgb(
                train_df, neg_valid_df, valid_df, fold, target_col, features
            )
        elif self.model_type == "lgbm_cls":
            valid_pred = self.train_lgbm_cls(
                train_df, neg_valid_df, valid_df, fold, target_col, features
            )
        elif self.model_type == "catboost":
            valid_pred = self.train_catboost(
                train_df, neg_valid_df, valid_df, fold, target_col, features
            )
        elif self.model_type == "catboost_ranker":
            valid_pred = self.train_catboost_ranker(
                train_df, neg_valid_df, valid_df, fold, target_col, features
            )
        elif self.model_type == "xgb":
            valid_pred = self.train_xgb(
                train_df, neg_valid_df, valid_df, fold, target_col, features
            )
        else:
            raise Exception(f"Unknown model_type: {self.model_type}")

        del train_df, neg_valid_df
        gc.collect()

        if self.config.is_dev:
            sessions = valid_df["session"].tolist()
            aids = valid_df["aid"].tolist()
            return list(
                zip(
                    sessions,
                    aids,
                    valid_pred.tolist(),
                )
            )
        return []

    def get_pred_dfs(
        self, pred_score_dict: Dict[str, np.ndarray]
    ) -> Dict[str, pd.DataFrame]:
        pred_dfs = {}
        for target_col in pred_score_dict.keys():
            # Result to DataFrame
            pred_df = (
                pl.DataFrame(
                    pred_score_dict[target_col],
                    columns=["session", "aid", "pred_score"],
                )
                .with_columns(
                    [
                        pl.col("session").cast(pl.Int32),
                        pl.col("aid").cast(pl.Int32),
                    ]
                )
                .sort(["session", "pred_score"], reverse=True)
                .groupby("session")
                .head(20)
                .groupby("session")
                .agg(pl.col("aid"))
                .with_column(pl.col("aid").alias("y_pred"))
                .select(
                    [
                        pl.col("session"),
                        pl.col("y_pred"),
                    ]
                )
                .to_pandas()
            )
            pred_df.to_parquet(self.output_dir / f"pred_{target_col}.parquet")
            pred_df["y_pred"] = pred_df["y_pred"].map(lambda x: x[:20])
            pred_dfs[target_col] = pred_df
        return pred_dfs

    def train_lgb(self, train_df, valid_df, pred_valid_df, fold, target_col, features):
        if self.config.is_dev:
            n_estimators = {
                "target_inter": 400,
                "target_buy": 300,
                "target_click": 400,
                "target_cart": 250,
                "target_order": 250,
            }
        else:
            n_estimators = {
                "target_inter": 900,
                "target_buy": 600,
                "target_click": 800,
                # "target_cart": 500,
                "target_cart": 600,
                "target_order": 500,
            }

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "verbosity": -1,
            "boosting": "gbdt",
            "is_unbalance": True,
            "seed": 77,
            "learning_rate": 0.05,
            "colsample_bytree": 0.5,
            "subsample_freq": 3,
            "subsample": 0.9,
            "n_estimators": 1000,
            "importance_type": "gain",
            "reg_lambda": 1.5,
            "reg_alpha": 0.1,
            "max_depth": 6,
            "num_leaves": 45,
            "eval_at": [20],
        }

        params["n_estimators"] = n_estimators[target_col]

        model = lgb.LGBMRanker(**params)

        train_baskets = train_df.groupby("session").size().values
        valid_baskets = valid_df.groupby("session").size().values

        model.fit(
            train_df[features],
            train_df[target_col],
            group=train_baskets,
            eval_group=[valid_baskets],
            eval_set=[(valid_df[features], valid_df[target_col])],
            callbacks=[
                # lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=30),
            ],
        )
        model.booster_.save_model(
            str(self.output_dir / f"lgbm_{target_col}_fold{fold}.txt")
        )
        if self.config.is_dev:
            return model.predict(pred_valid_df[features])
        return np.array([])

    def train_lgbm_cls(
        self, train_df, valid_df, pred_valid_df, fold, target_col, features
    ):
        if self.config.is_dev:
            n_estimators = {
                "target_inter": 400,
                "target_buy": 300,
                "target_click": 600,
                "target_cart": 400,
                "target_order": 400,
            }
        else:
            n_estimators = {
                "target_inter": 800,
                "target_buy": 500,
                "target_click": 800,
                "target_cart": 500,
                "target_order": 500,
            }

        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting": "gbdt",
            "is_unbalance": True,
            "seed": 77,
            "learning_rate": 0.005,
            "colsample_bytree": 0.5,
            "subsample_freq": 3,
            "subsample": 0.9,
            "n_estimators": 10000,
            "importance_type": "gain",
            "reg_lambda": 1.5,
            "reg_alpha": 0.1,
            "max_depth": 6,
            "num_leaves": 100,
        }

        params["n_estimators"] = n_estimators[target_col]

        model = lgb.LGBMClassifier(**params)

        model.fit(
            train_df[features],
            train_df[target_col],
            eval_set=[(valid_df[features], valid_df[target_col])],
            callbacks=[
                # lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=30),
            ],
        )
        model.booster_.save_model(
            str(self.output_dir / f"lgbm_cls_{target_col}_fold{fold}.txt")
        )
        if self.config.is_dev:
            return model.predict_proba(pred_valid_df[features])[:, 1]
        return np.array([])

    def train_catboost(
        self, train_df, valid_df, pred_valid_df, fold, target_col, features
    ):
        if self.config.is_dev:
            n_estimators = {
                "target_inter": 10000,
                "target_buy": 3000,
                "target_click": 10000,
                "target_cart": 3000,
                "target_order": 3000,
            }
        else:
            n_estimators = {
                "target_inter": 45000,
                "target_buy": 50000,
                "target_click": 40000,
                "target_cart": 20000,
                "target_order": 50000,
            }
        params = {
            "learning_rate": 0.01,
            "max_depth": 6,
            "random_state": 77,
            "thread_count": 2,
            "task_type": "GPU",
            "num_boost_round": 100000,
        }
        params["loss_function"] = "Logloss"
        if target_col == "target_click":
            params["scale_pos_weight"] = 100
            params["max_depth"] = 6
        else:
            params["scale_pos_weight"] = 200
            params["max_depth"] = 5

        params["num_boost_round"] = n_estimators[target_col]
        train_ds = Pool(
            train_df[features], train_df[target_col], group_id=train_df["session"]
        )
        valid_ds = Pool(
            valid_df[features], valid_df[target_col], group_id=valid_df["session"]
        )

        model = CatBoost(params)

        model.fit(
            train_ds,
            eval_set=valid_ds,
            use_best_model=True,
            # early_stopping_rounds=100,
            verbose_eval=200,
        )
        model.save_model(str(self.output_dir / f"catboost_{target_col}_fold{fold}.txt"))
        if self.config.is_dev:
            return model.predict(
                pred_valid_df[features], prediction_type="Probability"
            ).T[1]
        return np.array([])

    def train_catboost_ranker(
        self, train_df, valid_df, pred_valid_df, fold, target_col, features
    ):
        if self.config.is_dev:
            n_estimators = {
                "target_inter": 10000,
                "target_buy": 3000,
                "target_click": 10000,
                "target_cart": 3000,
                "target_order": 3000,
            }
        else:
            n_estimators = {
                "target_inter": 40000,
                "target_buy": 50000,
                "target_click": 40000,
                "target_cart": 20000,
                "target_order": 50000,
            }
        params = {
            "learning_rate": 0.01,
            "max_depth": 6,
            "random_state": 77,
            "thread_count": 2,
            "task_type": "GPU",
            "num_boost_round": 100000,
        }
        params["loss_function"] = "YetiRank"
        params["num_boost_round"] = n_estimators[target_col]
        if target_col == "target_click":
            params["max_depth"] = 6
        else:
            params["max_depth"] = 5

        train_ds = Pool(
            train_df[features], train_df[target_col], group_id=train_df["session"]
        )
        valid_ds = Pool(
            valid_df[features], valid_df[target_col], group_id=valid_df["session"]
        )

        model = CatBoost(params)

        model.fit(
            train_ds,
            eval_set=valid_ds,
            use_best_model=True,
            early_stopping_rounds=100,
            verbose_eval=200,
        )
        model.save_model(
            str(self.output_dir / f"catboost_ranker_{target_col}_fold{fold}.txt")
        )
        if self.config.is_dev:
            return model.predict(pred_valid_df[features])
        return np.array([])

    def train_xgb(self, train_df, valid_df, pred_valid_df, fold, target_col, features):
        if self.config.is_dev:
            n_estimators = {
                "target_inter": 600,
                "target_buy": 500,
                "target_click": 500,
                "target_cart": 250,
                "target_order": 400,
            }
        else:
            n_estimators = {
                "target_inter": 900,
                "target_buy": 800,
                "target_click": 800,
                "target_cart": 350,
                "target_order": 700,
            }

        xgb_params = {
            "tree_method": "gpu_hist",
            # "objective": "rank:ndcg",
            "objective": "rank:pairwise",
            "n_estimators": 1000,
            "eval_metric": "ndcg",
            # "max_depth": 10,
            "max_depth": 5,
            "eta": 0.05,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_lambda": 2,
            "alpha": 0.1,
            # "early_stopping_rounds": 100,
        }

        train_baskets = train_df.groupby("session").size().values
        valid_baskets = valid_df.groupby("session").size().values

        xgb_params["n_estimators"] = n_estimators[target_col]
        model = xgb.XGBRanker(**xgb_params)

        model.fit(
            train_df[features],
            train_df[target_col],
            group=train_baskets,
            eval_group=[valid_baskets],
            eval_set=[(valid_df[features], valid_df[target_col])],
            verbose=30,
        )

        model.save_model(str(self.output_dir / f"xgb_{target_col}_fold{fold}.txt"))
        if self.config.is_dev:
            return model.predict(pred_valid_df[features])
        return np.array([])

    def get_train_df(self, target_col, fold, cols) -> pd.DataFrame:
        return (
            self.filter_by_train_fold(
                self.add_targets(
                    self.filter_no_target_rows(
                        self.neg_sampler.scan(), target_col
                    )
                ),
                fold,
            )
            .sort("session")
            .select(cols)
            .collect()
            .to_pandas()
        )

    def get_valid_df(self, target_col, fold, cols) -> pd.DataFrame:
        if self.config.is_dev:
            return (
                self.add_targets(
                    self.filter_no_target_rows(
                        self.feature_merger.scan(), target_col
                    )
                )
                .filter(pl.col("fold") == fold)
                .sort("session")
                .select(cols)
                .collect()
                .to_pandas()
            )
        return pd.DataFrame()

    def get_neg_valid_df(self, target_col, fold, cols) -> pd.DataFrame:
        return (
            self.add_targets(
                self.filter_no_target_rows(
                    self.neg_sampler.scan(), target_col
                )
            )
            .filter(pl.col("fold") == fold)
            .sort("session")
            .select(cols)
            .collect()
            .to_pandas()
        )

    def add_targets(self, cand_df_lz: pl.LazyFrame) -> pl.LazyFrame:
        return cand_df_lz.with_columns(
            [
                # target_inter
                (
                    (pl.col("target_click") == 1)
                    | (pl.col("target_cart") == 1)
                    | (pl.col("target_order") == 1)
                )
                .cast(pl.Int8)
                .alias("target_inter"),
                # target_buy
                ((pl.col("target_cart") == 1) | (pl.col("target_order") == 1))
                .cast(pl.Int8)
                .alias("target_buy"),
            ]
        )

    def filter_no_target_rows(
        self, cand_df_lz: pl.LazyFrame, target_col: str
    ) -> pl.LazyFrame:
        if target_col in ["target_click", "target_cart", "target_order"]:
            return cand_df_lz.filter((pl.col(f"no_{target_col}") == 0))
        if target_col == "target_buy":
            return cand_df_lz.filter(
                (pl.col("no_target_cart") == 0) | (pl.col("target_order") == 1)
            )
        return cand_df_lz

    def filter_by_train_fold(self, cand_df_lz: pl.LazyFrame, fold: str) -> pl.LazyFrame:
        if not self.config.is_dev and (
            self.model_type == "lgbm" or self.model_type == "lgbm_cls"
        ):
            return cand_df_lz
        return cand_df_lz.filter(pl.col("fold") != fold)
