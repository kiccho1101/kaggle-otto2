from pathlib import Path

import polars as pl
import torch
import torch.backends
from torch.utils.data import DataLoader
from tqdm import tqdm

from kaggle_otto2.cand_generator.item_mf.dataset import ItemMFDataset
from kaggle_otto2.cand_generator.item_mf.model import ItemMFModel
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.util import EvaluateUtil, TimeUtil


class ItemMFTrainer:
    def __init__(self, data_loader: OttoDataLoader, config: Config):
        self.data_loader = data_loader
        self.config = config

        # parameters
        self.n_epochs = self.config.yaml.cg.item_mf.get("n_epochs", 15)
        self.n_factors = self.config.yaml.cg.item_mf.get("n_factors", 128)
        self.lr = self.config.yaml.cg.item_mf.get("lr", 0.0005)
        self.train_batch_size = self.config.yaml.cg.item_mf.get(
            "train_batch_size", 2**16
        )
        self.gen_cand_topk = self.config.yaml.cg.item_mf.get("gen_cand_topk", 40)

        self.model = ItemMFModel(len(self.data_loader.get_aid2idx()), self.n_factors)

    def fit(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True, parents=True)
        with TimeUtil.timer("load pair_df"):
            pair_df = self.data_loader.get_pair_df()
        print("pair_df:", pair_df.shape)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.n_epochs
        )

        train_ds = ItemMFDataset(self.data_loader)
        train_dl = DataLoader(
            train_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.best_score = -10e18
        for epoch in range(self.n_epochs):
            self.train(train_dl, epoch)
            if self.config.is_sub:
                self.save_model(data_dir)
            self.test(epoch)

    def train(self, train_dl, epoch):
        device = "cuda"
        self.model.train()
        self.model.to(device)
        with tqdm(train_dl, unit="batch", dynamic_ncols=True) as tepoch:
            for step, batch in enumerate(tepoch):
                tepoch.set_description(f"Train Epoch {epoch}")
                aid_x = batch["aid_x"].to(device)
                aid_y = batch["aid_y"].to(device)
                coef_x = batch["coef_x"].to(device)
                coef_y = batch["coef_y"].to(device)
                self.optimizer.zero_grad()

                loss = self.model.calc_loss(aid_x, aid_y, coef_x, coef_y)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                tepoch.set_postfix(loss=loss.item())

    def test(self, epoch):
        pred_df = (
            self.model.predict(self.data_loader)
            .sort(["session", "score"], reverse=[False, True])
            .groupby("session")
            .agg(pl.col("aid").alias("y_pred"))
        )
        pred_df.write_parquet(self.data_dir / "pred_df.parquet")
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
        self.model.load_state_dict(torch.load(data_dir / "model.pt"))
