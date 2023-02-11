import click

from kaggle_otto2.cand_generator import (
    Item2VecCandGenerator,
    ItemCFCandGenerator,
    ItemMFCandGenerator,
    LastInterCandGenerator,
    UserMFCandGenerator,
)
from kaggle_otto2.cand_merger import CandMerger
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader


@click.command()
@click.option("--exp", required=True, type=str)
def main(exp: str):
    config = Config(exp)
    data_loader = OttoDataLoader(config)

    common_params = [config.dir_config.exp_output_dir, config, data_loader]
    itemcf_params = [
        [[0, 1, 2], [0, 1, 2], True, True, False, True, False],
        [[1, 2], [1, 2], True, True, False, True, False],
        [[0], [0], True, True, False, True, False],
        [[0], [1, 2], True, True, False, True, False],
        [[0], [0, 1, 2], True, True, False, True, False],
        [[1, 2], [0, 1, 2], True, True, False, True, False],
        [[0, 1, 2], [0, 1, 2], True, True, True, True, True],
        [[1, 2], [0, 1, 2], True, True, True, True, True],
        [[0, 1, 2], [0, 1, 2], False, True, True, True, False],
    ]
    itemcf_agg_methods = ["sum", "last", "count"]
    cand_generators = [
        # LastInter
        LastInterCandGenerator(*[*common_params, "inter"]),
        LastInterCandGenerator(*[*common_params, "buy"]),
        LastInterCandGenerator(*[*common_params, "click"]),
        LastInterCandGenerator(*[*common_params, "cart"]),
        LastInterCandGenerator(*[*common_params, "order"]),
        # ItemCF
        *[
            ItemCFCandGenerator(*[*common_params, agg_method, *itemcf_param])
            for itemcf_param in itemcf_params
            for agg_method in itemcf_agg_methods
        ],
        # ItemMF
        ItemMFCandGenerator(*[*common_params, "last"]),
        ItemMFCandGenerator(*[*common_params, "seq"]),
        # UserMF
        UserMFCandGenerator(*common_params),
        # # Item2Vec
        # Item2VecCandGenerator(*[*common_params, "last"]),
        # Item2VecCandGenerator(*[*common_params, "seq"]),
    ]
    cand_merger = CandMerger(config.dir_config.exp_output_dir, config, data_loader)
    cand_merger.merge(cand_generators)
    cand_merger.calc_score()


if __name__ == "__main__":
    main()
