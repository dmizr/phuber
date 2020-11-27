import hydra
from omegaconf import DictConfig

from phuber.train import train


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    run()
