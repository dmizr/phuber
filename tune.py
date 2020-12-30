import hydra
from omegaconf import DictConfig

import phuber.runner as runner
import phuber.utils as utils


@hydra.main(config_path="conf", config_name="tune")
def tune(cfg: DictConfig):
    utils.display_config(cfg)
    train_acc, val_acc, test_acc = runner.train(cfg)

    return val_acc


if __name__ == "__main__":
    tune()
