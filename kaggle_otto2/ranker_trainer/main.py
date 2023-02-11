import click

from kaggle_otto2.cand_merger import NegSampler
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import FeatureMerger
from kaggle_otto2.ranker_trainer import RankerTrainer


@click.command()
@click.option("--exp", required=True, type=str)
@click.option("--model_type", required=True, type=str)
def main(exp: str, model_type: str):
    if model_type not in ["lgbm", "lgbm_cls", "xgb", "catboost", "catboost_ranker"]:
        raise ValueError(f"model_type {model_type} is not supported")

    config = Config(exp)
    data_loader = OttoDataLoader(config)

    feature_merger = FeatureMerger(
        config.dir_config.exp_output_dir, data_loader, config
    )
    neg_sampler = NegSampler(config.dir_config.exp_output_dir, data_loader, config)

    trainer = RankerTrainer(
        config.dir_config.exp_output_dir,
        data_loader,
        feature_merger,
        neg_sampler,
        config,
        model_type,
    )

    trainer.fit(config.yaml.ranker.get("features", []))


if __name__ == "__main__":
    main()
