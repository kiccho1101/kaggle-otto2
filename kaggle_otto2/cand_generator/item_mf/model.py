import numpy as np
import polars as pl
import torch
from torch import nn
from tqdm import tqdm

from kaggle_otto2.cand_generator.item_mf.enums import SeedType
from kaggle_otto2.cand_generator.item_mf.loss import BPRLoss
from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.util import SearchUtil, TimeUtil


class ItemMFModel(nn.Module):
    def __init__(self, n_aid: int, n_factors: int):
        super().__init__()
        self.criterion = BPRLoss()
        self.n_factors = n_factors
        self.n_aid = n_aid
        self.aid_embeddings = nn.Embedding(self.n_aid, self.n_factors)

        initrange = 1.0 / self.n_factors
        nn.init.uniform_(self.aid_embeddings.weight.data, -initrange, initrange)

    def forward(self, aid_x, aid_y, coef_x, coef_y):
        aid_x = self.aid_embeddings(aid_x)
        aid_y = self.aid_embeddings(aid_y)
        return (aid_x * aid_y).sum(dim=1) * coef_x * coef_y

    def calc_loss(self, aid_x, aid_y, coef_x, coef_y):
        rand_idx = torch.randperm(aid_y.size(0))
        output_pos = self.forward(aid_x, aid_y, coef_x, coef_y)
        output_neg = self.forward(aid_x, aid_y[rand_idx], coef_x, coef_y[rand_idx])
        loss = self.criterion(output_pos, output_neg)
        return loss

    def predict(
        self, data_loader: OttoDataLoader, topk=20, seed_type: SeedType = "last"
    ) -> pl.DataFrame:
        self.eval()
        embeddings = self.aid_embeddings.weight.data.cpu().numpy()
        idx2aid = data_loader.get_idx2aid()
        with TimeUtil.timer("prepare data"):
            if seed_type == "seq":
                # 直近の履歴を重視するように重み付け
                # devで実験してこれが一番良かった
                weights = np.linspace(1.0, 0.2, 17)
                k = len(weights)
                with TimeUtil.timer("get test_aids"):
                    # test sessionごとにitemのlistを取得
                    test_aids_df = (
                        data_loader.get_test_df()
                        .sort(["session", "ts"], reverse=[False, True])
                        .join(data_loader.get_aid_idx_df(), on="aid")
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
                    data_loader.get_test_df()
                    .sort(["session", "ts"], reverse=[False, True])
                    .groupby("session")
                    .head(1)  # Get last item in test session
                    .join(data_loader.get_aid_idx_df(), on="aid")
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
                    pred_aids.append(idx2aid[aids[i][j]])
                    pred_scores.append(distances[i][j])

        return pl.DataFrame(
            {"session": pred_sessions, "aid": pred_aids, "score": pred_scores}
        )
