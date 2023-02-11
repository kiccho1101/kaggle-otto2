import polars as pl
import torch
from torch import nn
from tqdm import tqdm

from kaggle_otto2.cand_generator.user_mf.loss import BPRLoss
from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.util import SearchUtil, TimeUtil


class UserMFModel(nn.Module):
    def __init__(self, n_session: int, n_aid: int, n_factors: int):
        super().__init__()
        self.n_factors = n_factors
        self.n_session = n_session
        self.n_aid = n_aid

        self.session_embeddings = nn.Embedding(
            self.n_session,
            self.n_factors,
            dtype=torch.float32,
        )
        self.aid_embeddings = nn.Embedding(
            self.n_aid, self.n_factors, dtype=torch.float32
        )

        self.criterion = BPRLoss()

        initrange = 1.0 / self.n_factors
        nn.init.uniform_(self.session_embeddings.weight.data, -initrange, initrange)
        nn.init.uniform_(self.aid_embeddings.weight.data, -initrange, initrange)

    def set_aid_embeddings_grad(self, requires_grad: bool):
        for param in self.aid_embeddings.parameters():
            param.requires_grad = requires_grad

    def forward(self, session, aid, aid_size):
        session_emb = self.session_embeddings(session)
        aid_emb = self.aid_embeddings(aid)
        return (session_emb * aid_emb).sum(dim=1) * aid_size

    def calc_loss(self, session, aid, aid_size):
        rand_idx = torch.randperm(aid.size(0))
        output_pos = self.forward(session, aid, aid_size)
        output_neg = self.forward(session, aid[rand_idx], aid_size[rand_idx])

        loss = self.criterion(output_pos, output_neg)

        return loss

    def predict(
        self, data_loader: OttoDataLoader, idx2aid, session2idx, topk=20
    ) -> pl.DataFrame:
        self.eval()

        aid_embeddings = self.aid_embeddings.weight.data.cpu().numpy()
        session_embeddings = self.session_embeddings.weight.data.cpu().numpy()

        test_df = data_loader.get_test_df()
        sessions = test_df["session"].unique().to_numpy()
        sessions_idx = [session2idx[s] for s in sessions]

        query_vectors = session_embeddings[sessions_idx]

        pred_sessions = []
        pred_aids = []
        pred_scores = []
        with TimeUtil.timer("search"):
            distances, aids = SearchUtil.ann_search(aid_embeddings, query_vectors, topk)
            for i in tqdm(range(aids.shape[0])):
                for j in range(aids.shape[1]):
                    pred_sessions.append(sessions[i])
                    pred_aids.append(idx2aid[aids[i][j]])
                    pred_scores.append(distances[i][j])

        return pl.DataFrame(
            {"session": pred_sessions, "aid": pred_aids, "score": pred_scores}
        )
