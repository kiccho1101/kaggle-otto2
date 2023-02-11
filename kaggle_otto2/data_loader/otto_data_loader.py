import gc
from typing import Dict, Tuple

import numpy as np
import polars as pl
from tqdm import tqdm

from kaggle_otto2.config import Config
from kaggle_otto2.util import FileUtil, TimeUtil


class OttoDataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.data_dir = self.config.dir_config.exp_output_dir / "data_loader"
        self.data_dir.mkdir(exist_ok=True, parents=True)

    def fit(self):
        self.create_train_test_df()
        self.create_idx_maps()
        self.create_pair_df()

    def create_train_test_df(self):
        with TimeUtil.timer("create train_df, test_df, test_labels"):
            with TimeUtil.timer("load"):
                input_dir = (
                    self.config.dir_config.input_dir
                    / "otto-full-optimized-memory-footprint"
                )
                if self.config.is_cv:
                    input_dir = (
                        self.config.dir_config.input_dir
                        / "otto-train-and-test-data-for-local-validation"
                    )
                test_df = pl.read_parquet(input_dir / "test.parquet").with_columns(
                    [
                        pl.col("session").cast(pl.Int32),
                        pl.col("aid").cast(pl.Int32),
                    ]
                )
                train_df = pl.read_parquet(input_dir / "train.parquet").with_columns(
                    [
                        pl.col("session").cast(pl.Int32),
                        pl.col("aid").cast(pl.Int32),
                    ]
                )
                train_sessions = train_df["session"].unique().to_numpy()
                train_df = pl.concat([train_df, test_df])

                test_labels = pl.DataFrame()
                if self.config.is_cv:
                    type_map = {"clicks": 0, "carts": 1, "orders": 2}
                    test_labels = (
                        pl.read_parquet(input_dir / "test_labels.parquet")
                        .with_columns(
                            [
                                pl.col("session").cast(pl.Int32),
                                pl.col("ground_truth").cast(pl.List(pl.Int32)),
                            ]
                        )
                        .with_columns(
                            pl.col("type").apply(lambda x: type_map[x]).cast(pl.UInt8)
                        )
                    )

            if self.config.is_dev:
                train_df, test_df, test_labels = self.sampling(
                    train_sessions, train_df, test_df, test_labels
                )

        with TimeUtil.timer("save"):
            train_df.write_parquet(self.data_dir / "train_df.parquet")
            test_df.write_parquet(self.data_dir / "test_df.parquet")
            test_labels.write_parquet(self.data_dir / "test_labels.parquet")
            del train_df, test_df, test_labels
            gc.collect()

    def create_idx_maps(self):
        train_df = self.get_train_df()

        with TimeUtil.timer("create aid idx map"):
            aids = train_df["aid"].unique().to_numpy()
            aid_idxs = np.arange(len(aids))
            aid_idx_df = pl.DataFrame(
                {"aid": aids, "aid_idx": aid_idxs},
                schema={"aid": pl.Int32, "aid_idx": pl.Int32},
            )
            aid2idx = dict(zip(aids, aid_idxs))
            aid_idx_df.write_parquet(self.data_dir / "aid_idx_df.parquet")
            FileUtil.save_pickle(aid2idx, self.data_dir / "aid2idx.pkl")
            FileUtil.save_pickle(
                {k: v for k, v in enumerate(aids)}, self.data_dir / "idx2aid.pkl"
            )

        with TimeUtil.timer("create session idx map"):
            sessions = train_df["session"].unique().to_numpy()
            session_idxs = np.arange(len(sessions))
            session_idx_df = pl.DataFrame(
                {"session": sessions, "session_idx": session_idxs},
                schema={"session": pl.Int32, "session_idx": pl.Int32},
            )
            session2idx = dict(zip(sessions, session_idxs))
            session_idx_df.write_parquet(self.data_dir / "session_idx_df.parquet")
            FileUtil.save_pickle(session2idx, self.data_dir / "session2idx.pkl")
            FileUtil.save_pickle(
                {k: v for k, v in enumerate(sessions)}, self.data_dir / "idx2session.pkl"
            )

    def create_pair_df(self):
        with TimeUtil.timer("create pair_df"):
            train_df = self.get_train_df()
            aid_idx_df = self.get_aid_idx_df()
            train_df = (
                train_df.join(aid_idx_df, on="aid")
                .drop(["aid", "type"])
                .sort(["session", "ts"], reverse=[False, True])
            )
            pair_df = pl.DataFrame()
            for i in tqdm(range(1, 5)):
                _pair_df = (
                    train_df.with_columns(
                        [
                            pl.col("aid_idx")
                            .shift(-i)
                            .over("session")
                            .alias("next_aid_idx"),
                            (
                                pl.col("ts") - pl.col("ts").shift(-i).over("session")
                            ).alias("ts_diff"),
                        ]
                    )
                    .filter(pl.col("next_aid_idx").is_not_null())
                    .select(["aid_idx", "next_aid_idx", "ts_diff"])
                )
                pair_df = pl.concat([pair_df, _pair_df])

        print("pair_df:", pair_df.shape)

        with TimeUtil.timer("save pair_df"):
            pair_df.write_parquet(self.data_dir / "pair_df.parquet")

    def sampling(
        self,
        train_sessions,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        test_labels: pl.DataFrame,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        sampling_ratio = self.config.yaml.pp.get("sampling_ratio", 0.05)
        with TimeUtil.timer(f"sampling (ratio: {sampling_ratio})"):
            test_sessions = test_df["session"].unique()

            sampled_train_sessions = set(
                np.random.choice(
                    train_sessions,
                    int(len(train_sessions) * sampling_ratio),
                )
            )
            sampled_test_sessions = set(
                np.random.choice(
                    test_sessions,
                    int(len(test_sessions) * sampling_ratio),
                )
            )
            sampled_train_sessions = sampled_train_sessions.union(sampled_test_sessions)

            train_df = train_df.filter(
                pl.col("session").is_in(list(sampled_train_sessions))
            )
            test_df = test_df.filter(
                pl.col("session").is_in(list(sampled_test_sessions))
            )
            test_labels = test_labels.filter(
                pl.col("session").is_in(list(sampled_test_sessions))
            )

            gc.collect()

        return train_df, test_df, test_labels

    def get_train_df(self) -> pl.DataFrame:
        return pl.read_parquet(self.data_dir / "train_df.parquet")

    def get_test_df(self) -> pl.DataFrame:
        return pl.read_parquet(self.data_dir / "test_df.parquet")

    def get_test_labels(self) -> pl.DataFrame:
        return pl.read_parquet(self.data_dir / "test_labels.parquet")

    def get_aid_idx_df(self) -> pl.DataFrame:
        return pl.read_parquet(self.data_dir / "aid_idx_df.parquet")

    def get_aid2idx(self) -> Dict[int, int]:
        return FileUtil.load_pickle(self.data_dir / "aid2idx.pkl")

    def get_idx2aid(self) -> Dict[int, int]:
        return FileUtil.load_pickle(self.data_dir / "idx2aid.pkl")

    def get_session_idx_df(self) -> pl.DataFrame:
        return pl.read_parquet(self.data_dir / "session_idx_df.parquet")

    def get_session2idx(self) -> Dict[int, int]:
        return FileUtil.load_pickle(self.data_dir / "session2idx.pkl")

    def get_idx2session(self) -> Dict[int, int]:
        return FileUtil.load_pickle(self.data_dir / "idx2session.pkl")

    def get_pair_df(self) -> pl.DataFrame:
        return pl.read_parquet(self.data_dir / "pair_df.parquet")


if __name__ == "__main__":
    config = Config("exp001_dev")
    data_loader = OttoDataLoader(config)

    data_loader.fit()

    train_df = data_loader.get_train_df()
    test_df = data_loader.get_test_df()
    test_labels = data_loader.get_test_labels()

    print(train_df)
    print(test_df)
    print(test_labels)

    assert train_df.shape == (8380130, 4)
    assert train_df.columns == ["session", "aid", "ts", "type"]
    assert train_df.dtypes == [pl.Int32, pl.Int32, pl.Int32, pl.UInt8]

    assert test_df.shape == (377192, 4)
    assert test_df.columns == ["session", "aid", "ts", "type"]
    assert test_df.dtypes == [pl.Int32, pl.Int32, pl.Int32, pl.UInt8]

    assert test_labels.shape == (107950, 3)
    assert test_labels.columns == ["session", "type", "ground_truth"]
    assert test_labels.dtypes == [pl.Int32, pl.UInt8, pl.List(pl.Int32)]
