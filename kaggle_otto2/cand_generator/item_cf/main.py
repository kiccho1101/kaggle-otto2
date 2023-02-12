import click

from kaggle_otto2.cand_generator import ItemCFCandGenerator
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader


@click.command()
@click.option("--exp", required=True, type=str)
def main(exp: str):
    config = Config(exp)
    data_loader = OttoDataLoader(config)

    params = [
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
    for param in params:
        agg_method = "sum"
        cg = ItemCFCandGenerator(
            config.dir_config.exp_output_dir, data_loader, config, agg_method, *param
        )
        cg.fit()

    for param in params:
        for agg_method in ["sum", "mean", "max", "last", "count"]:
            cg = ItemCFCandGenerator(
                config.dir_config.exp_output_dir,
                data_loader,
                config,
                agg_method,
                *param
            )
            cg.gen_cand_df()
            if config.is_cv:
                cg.calc_score()


if __name__ == "__main__":
    main()
