import click
from tqdm import tqdm

from kaggle_otto2.cand_merger import CandMerger, NegSampler
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader
from kaggle_otto2.feature import (FeatureMerger, ItemInterNum, ItemInterRank,
                                  ItemInterTsStats, ItemMultiInterProb,
                                  ItemTypeToTypeProb, UserInterDow,
                                  UserInterNum, UserInterTsDiff,
                                  UserInterTsHour, UserInterTsStats,
                                  UserItemFirstInterRank, UserItemIsFirstInter,
                                  UserTsDiff)
from kaggle_otto2.util import TimeUtil


@click.command()
@click.option("--exp", required=True, type=str)
def main(exp: str):
    config = Config(exp)
    data_loader = OttoDataLoader(config)

    common_params = [config.dir_config.exp_output_dir, data_loader]
    features = [
        ItemInterNum(*common_params),
        ItemInterRank(*common_params),
        ItemInterTsStats(*common_params),
        ItemMultiInterProb(*common_params),
        ItemTypeToTypeProb(*common_params),
        UserInterDow(*common_params),
        UserInterNum(*common_params),
        UserInterTsDiff(*common_params),
        UserInterTsHour(*common_params),
        UserInterTsStats(*common_params),
        UserItemFirstInterRank(*common_params),
        UserItemIsFirstInter(*common_params),
        UserTsDiff(*common_params),
    ]

    with TimeUtil.timer("create features"):
        for feature in tqdm(features):
            feature.fit()

    with TimeUtil.timer("merge features"):
        cand_merger = CandMerger(config.dir_config.exp_output_dir, config, data_loader)
        feature_merger = FeatureMerger(
            config.dir_config.exp_output_dir, data_loader, config
        )
        feature_merger.merge(cand_merger, features)

    with TimeUtil.timer("negative sampling"):
        feature_merger = FeatureMerger(
            config.dir_config.exp_output_dir, data_loader, config
        )
        neg_sampler = NegSampler(config.dir_config.exp_output_dir, data_loader, config)
        neg_sampler.neg_sampling(feature_merger)


if __name__ == "__main__":
    main()
