import sys

from hydra import compose, initialize_config_dir

from kaggle_otto2.config.dir_config import DirConfig
from kaggle_otto2.config.enums import EnvEnum
from kaggle_otto2.util.global_util import GlobalUtil


class Config:
    def __init__(self, exp: str):
        self.exp = exp
        self._init_config()

    def _init_config(self):
        self.env = self._get_env()
        self.dir_config = DirConfig(self.exp, self.env)
        self._init_yaml()
        GlobalUtil.seed_everything(self.yaml.get("seed", 77))

        self.is_dev = "_dev" in self.exp
        self.is_cv = "_cv" in self.exp or "_dev" in self.exp
        self.is_sub = "_lb" in self.exp

    def _init_yaml(self):
        with initialize_config_dir(
            config_dir=str(self.dir_config.root_dir / "yaml"), version_base=None
        ):
            self.yaml = compose(config_name=self.exp)

    def _get_env(self) -> EnvEnum:
        if "kaggle_gcp" in sys.modules:
            return "kaggle"
        if "google.colab" in sys.modules:
            return "colab"
        return "local"


if __name__ == "__main__":
    config = Config("exp001_dev")

    print(config.env)
    print(config.exp)
    print(config.yaml)
