import hydra
from omegaconf import DictConfig

import phuber.runner as runner
import phuber.utils as utils


@hydra.main(config_path="conf", config_name="train")
def train(cfg: DictConfig):
    utils.display_config(cfg)
    runner.train(cfg)


if __name__ == "__main__":
    train()
