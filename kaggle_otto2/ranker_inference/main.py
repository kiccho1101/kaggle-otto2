import click

from kaggle_otto2.cand_merger import NegSampler
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import FeatureMerger
from kaggle_otto2.ranker_inference import RankerInference


@click.command()
@click.option("--exp", required=True, type=str)
def main(exp: str):
    config = Config(exp)
    data_loader = OttoDataLoader(config)

    feature_merger = FeatureMerger(
        config.dir_config.exp_output_dir, data_loader, config
    )
    neg_sampler = NegSampler(config.dir_config.exp_output_dir, data_loader, config)

    ranker_inference = RankerInference(
        config.dir_config.exp_output_dir,
        data_loader,
        feature_merger,
        neg_sampler,
        config,
        # model_types=["lgbm", "lgbm_cls", "xgb", "catboost", "catboost_ranker"],
        model_types=["lgbm"],
        targets=[
            "target_inter",
            "target_buy",
            "target_click",
            "target_cart",
            "target_order",
        ]
    )
    ranker_inference.inference(config.yaml.ranker.get("features", []))


if __name__ == "__main__":
    main()
