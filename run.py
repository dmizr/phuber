import hydra
from omegaconf import DictConfig, OmegaConf
import phuber
from phuber.train import train


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    phuber.train.train(cfg)


if __name__ == "__main__":
    run()
