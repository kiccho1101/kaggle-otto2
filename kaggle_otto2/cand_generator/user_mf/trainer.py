from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kaggle_otto2.cand_generator.user_mf.dataset import UserMFDataset
from kaggle_otto2.cand_generator.user_mf.model import UserMFModel
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.util import EvaluateUtil, FileUtil, TimeUtil


class UserMFTrainer:
    def __init__(self, data_loader: OttoDataLoader, config: Config):
        self.data_loader = data_loader
        self.config = config

        # parameters
        self.n_epochs = self.config.yaml.cg.user_mf.get("n_epochs", 15)
        self.n_factors = self.config.yaml.cg.user_mf.get("n_factors", 128)
        self.lr = self.config.yaml.cg.user_mf.get("lr", 0.0005)
        self.train_batch_size = self.config.yaml.cg.user_mf.get(
            "train_batch_size", 65536
        )
        self.user_min_inters = self.config.yaml.cg.user_mf.get("user_min_inters", 20)
        self.gen_cand_topk = self.config.yaml.cg.user_mf.get("gen_cand_topk", 40)

    def fit(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True, parents=True)

        with TimeUtil.timer("load data"):
            train_df = self.data_loader.get_train_df()
            test_df = self.data_loader.get_test_df()

        with TimeUtil.timer(f"filter sessions with events >= {self.user_min_inters}"):
            test_sessions = test_df["session"].unique().to_list()
            target_sessions = (
                train_df.groupby("session")
                .agg(pl.count("aid").alias("n_events"))
                .filter(
                    (pl.col("n_events") >= self.user_min_inters)
                    | (pl.col("session")).is_in(test_sessions)
                )["session"]
                .to_list()
            )
            train_df = train_df.filter(pl.col("session").is_in(target_sessions))

        with TimeUtil.timer("get idx2aid, aid2idx, idx2session, session2idx"):
            idx2aid = train_df["aid"].unique().to_list()
            aid2idx = {aid: idx for idx, aid in enumerate(idx2aid)}
            idx2session = {idx: session for idx, session in enumerate(target_sessions)}
            session2idx = {session: idx for idx, session in enumerate(target_sessions)}
            FileUtil.save_pickle(idx2aid, self.data_dir / "idx2aid.pkl")
            FileUtil.save_pickle(aid2idx, self.data_dir / "aid2idx.pkl")
            FileUtil.save_pickle(idx2session, self.data_dir / "idx2session.pkl")
            FileUtil.save_pickle(session2idx, self.data_dir / "session2idx.pkl")
            n_session = len(session2idx.keys())
            n_aid = len(aid2idx.keys())
            FileUtil.save_pickle(n_aid, self.data_dir / "n_aid.pkl")
            FileUtil.save_pickle(n_session, self.data_dir / "n_session.pkl")

        with TimeUtil.timer("init model"):
            self.model = UserMFModel(
                n_session,
                n_aid,
                self.n_factors,
            )

        with TimeUtil.timer("get train_dl"):
            train_ds = UserMFDataset(train_df, session2idx, aid2idx)
            train_dl = DataLoader(
                train_ds,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=2,
                drop_last=True,
                pin_memory=True,
            )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.n_epochs
        )

        self.best_score = -10e18
        for epoch in range(self.n_epochs):
            self.train(epoch, train_dl)
            self.test(epoch)

    def train(self, epoch, train_dl):
        device = "cuda"
        self.model.train()
        self.model.to(device)
        with tqdm(train_dl, unit="batch", dynamic_ncols=True) as tepoch:
            for step, batch in enumerate(tepoch):
                tepoch.set_description(f"Train Epoch {epoch}")
                session = batch["session"].to(device)
                aid = batch["aid"].to(device)
                aid_size = batch["aid_size"].to(device)
                self.optimizer.zero_grad()

                loss = self.model.calc_loss(session, aid, aid_size)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                tepoch.set_postfix(loss=loss.item())
        torch.cuda.empty_cache()

    def test(self, epoch):
        self.model.eval()
        self.model.to("cpu")
        torch.cuda.empty_cache()
        idx2aid = FileUtil.load_pickle(self.data_dir / "idx2aid.pkl")
        session2idx = FileUtil.load_pickle(self.data_dir / "session2idx.pkl")
        pred_df = (
            self.model.predict(self.data_loader, idx2aid, session2idx)
            .sort(["session", "score"], reverse=[False, True])
            .groupby("session")
            .agg(pl.col("aid").alias("y_pred"))
        )
        pred_dfs = {
            "target_click": pred_df,
            "target_cart": pred_df,
            "target_order": pred_df,
        }
        score, scores, _ = EvaluateUtil.calc_score(
            self.data_loader.get_test_labels(), pred_dfs, topk=20, verbose=True
        )
        if score > self.best_score:
            self.best_score = score
            self.save_model(self.data_dir)
        print(f"==================== EPOCH {epoch} RESULT ====================")
        print("score: ", score)
        print("best : ", self.best_score)
        print("scores: ", scores)

    def save_model(self, data_dir: Path):
        torch.save(self.model.state_dict(), data_dir / "model.pt")

    def load_model(self, data_dir: Path):
        n_aid = FileUtil.load_pickle(data_dir / "n_aid.pkl")
        n_session = FileUtil.load_pickle(data_dir / "n_session.pkl")
        self.model = UserMFModel(
            n_session,
            n_aid,
            self.n_factors,
        )
        self.model.load_state_dict(torch.load(data_dir / "model.pt"))
