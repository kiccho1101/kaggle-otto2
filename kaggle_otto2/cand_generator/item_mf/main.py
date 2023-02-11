import click

from kaggle_otto2.cand_generator import ItemMFCandGenerator
from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader


@click.command()
@click.option("--exp", required=True, type=str)
def main(exp: str):
    config = Config(exp)
    data_loader = OttoDataLoader(config)

    cg = ItemMFCandGenerator(config.dir_config.exp_output_dir, config, data_loader)
    # cg.fit()
    for seed_type in ["last", "seq"]:
        cg = ItemMFCandGenerator(
            config.dir_config.exp_output_dir, config, data_loader, seed_type
        )
        cg.gen_cand_df()
        cg.calc_score()


if __name__ == "__main__":
    main()
