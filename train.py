from __future__ import absolute_import, division, print_function
import torch
import random
import numpy as np
from trainer import Trainer
from options import ECDepthOptions


def seed_all(seed):
    if not seed:
        seed = 1

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


options = ECDepthOptions()
opts = options.parse()
seed_all(opts.pytorch_random_seed)

if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()