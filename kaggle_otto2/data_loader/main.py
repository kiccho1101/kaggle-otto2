import click

from kaggle_otto2.config import Config
from kaggle_otto2.data_loader import OttoDataLoader


@click.command()
@click.option("--exp", required=True, type=str)
def main(exp: str):
    config = Config(exp)
    data_loader = OttoDataLoader(config)
    data_loader.fit()


if __name__ == "__main__":
    main()
