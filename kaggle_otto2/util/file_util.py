import pickle
from pathlib import Path
from typing import Any, Union

import cloudpickle


class FileUtil:
    @staticmethod
    def save_pickle(obj: Any, file_path: Union[str, Path], ignore_error: bool = False) -> None:
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                cloudpickle.dump(obj, f)
        except Exception as e:
            if ignore_error:
                print(e)
            else:
                raise e

    @staticmethod
    def load_pickle(file_path: Union[str, Path]) -> Any:
        with open(file_path, "rb") as f:
            return pickle.load(f)
