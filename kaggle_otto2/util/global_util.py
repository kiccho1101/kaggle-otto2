import os
import random
import time
from typing import Tuple

import numpy as np
import psutil
import torch
import torch.backends.cudnn
import torch.cuda


class GlobalUtil:
    @staticmethod
    def seed_everything(seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    @staticmethod
    def get_metric() -> Tuple[float, float, float]:
        t = time.time()
        p = psutil.Process(os.getpid())
        m: float = p.memory_info()[0] / 2.0**30
        per: float = psutil.virtual_memory().percent
        return t, m, per
