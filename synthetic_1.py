import hydra
from omegaconf import DictConfig

from synthetic.experiments import long_servedio_experiment


@hydra.main(config_path="conf", config_name="synthetic/long_servedio")
def synthetic_1(cfg: DictConfig) -> None:
    long_servedio_experiment(cfg)


if __name__ == "__main__":
    synthetic_1()
