import gc
import logging
from pathlib import Path

import numpy as np
import polars as pl
from gensim.models import Word2Vec
from tqdm import tqdm

from kaggle_otto2.cand_generator.item2vec.enums import SeedType
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.util import SearchUtil, TimeUtil

# Word2Vecの学習ログが出るようにloggingを設定
logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


class Item2VecTrainer:
    def __init__(self, data_loader: OttoDataLoader, config: Config):
        self.data_loader = data_loader
        self.config = config

        # parameters
        self.epochs = self.config.yaml.cg.item2vec.get("epochs", 10)
        self.vector_size = self.config.yaml.cg.item2vec.get("vector_size", 32)
        self.lr = self.config.yaml.cg.user_mf.get("lr", 0.1)
        self.window = self.config.yaml.cg.user_mf.get("window", 5)
        self.sg = self.config.yaml.cg.user_mf.get("sg", 1)
        self.negative = self.config.yaml.cg.user_mf.get("negative", 20)
        self.min_count = self.config.yaml.cg.user_mf.get("min_count", 1)
        self.user_min_inters = self.config.yaml.cg.user_mf.get("user_min_inters", 20)
        self.gen_cand_topk = self.config.yaml.cg.user_mf.get("gen_cand_topk", 40)

    def fit(self, model_dir: Path, workers=4):
        with TimeUtil.timer("get aid_sentences"):
            aid_sentences = (
                self.data_loader.get_train_df()
                .sort(["session", "ts"], reverse=[False, False])
                .groupby("session")
                .agg(pl.col("aid").cast(pl.Utf8))["aid"]
                .to_list()
            )

        with TimeUtil.timer("train item2vec"):
            item2vec = Word2Vec(
                aid_sentences,
                alpha=self.lr,
                window=self.window,
                vector_size=self.vector_size,
                sg=self.sg,
                epochs=self.epochs,
                seed=self.config.yaml.get("seed", 77),
                negative=self.negative,
                min_count=self.min_count,
                workers=workers,
            )
        item2vec.save(str(model_dir / "item2vec.model"))
        del item2vec
        gc.collect()

    def predict(self, model_dir: Path, topk=20, seed_type: SeedType = "last"):
        with TimeUtil.timer("load item2vec"):
            item2vec = Word2Vec.load(str(model_dir / "item2vec.model"))

        embeddings = item2vec.wv.vectors
        idx2aid = item2vec.wv.index_to_key

        with TimeUtil.timer("prepare data"):
            if seed_type == "seq":
                # 直近の履歴を重視するように重み付け
                # devで実験してこれが一番良かった
                weights = np.linspace(1.0, 0.2, 17)
                k = len(weights)
                with TimeUtil.timer("get test_aids"):
                    # test sessionごとにitemのlistを取得
                    test_aids_df = (
                        self.data_loader.get_test_df()
                        .sort(["session", "ts"], reverse=[False, True])
                        .join(self.data_loader.get_aid_idx_df(), on="aid")
                        .groupby("session")
                        .agg(pl.col("aid_idx"))
                    )
                    test_aids = dict(
                        zip(
                            test_aids_df["session"].to_list(),
                            test_aids_df["aid_idx"].to_list(),
                        )
                    )

                with TimeUtil.timer("get test_aids k"):
                    # test sessionごとにitemのlistをk個にする
                    for session in test_aids.keys():
                        test_aids[session] = np.tile(test_aids[session], k)
                        test_aids[session] = test_aids[session][:k]

                # aid_idxs(履歴のitemのindexのlist)を作成
                sessions = []
                aid_idxs = []
                for session in test_aids.keys():
                    sessions.append(session)
                    aid_idxs.append(test_aids[session])
                aid_idxs = np.array(aid_idxs)

                with TimeUtil.timer("get query_vectors"):
                    # 重み付け平均からquery_vectorsを作成
                    w = np.array(
                        [
                            np.array(weights).reshape(len(weights), 1)
                            for _ in range(len(aid_idxs))
                        ]
                    )
                    query_vectors = (embeddings[aid_idxs] * w).sum(axis=1)
            else:
                test_aids_df = (
                    self.data_loader.get_test_df()
                    .sort(["session", "ts"], reverse=[False, True])
                    .groupby("session")
                    .head(1)  # Get last item in test session
                    .join(self.data_loader.get_aid_idx_df(), on="aid")
                    .select(["session", "aid_idx"])
                )
                query_vectors = embeddings[test_aids_df["aid_idx"].to_list()]

        sessions = test_aids_df["session"].to_list()

        pred_sessions = []
        pred_aids = []
        pred_scores = []
        with TimeUtil.timer("search"):
            distances, aids = SearchUtil.ann_search(embeddings, query_vectors, topk)
            for i in tqdm(range(aids.shape[0])):
                for j in range(aids.shape[1]):
                    pred_sessions.append(sessions[i])
                    pred_aids.append(int(idx2aid[aids[i][j]]))
                    pred_scores.append(distances[i][j])

        return pl.DataFrame(
            {"session": pred_sessions, "aid": pred_aids, "score": pred_scores}
        )
