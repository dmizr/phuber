import hydra
from omegaconf import DictConfig

from synthetic.experiments import outliers_experiment


@hydra.main(config_path="conf", config_name="synthetic/outliers")
def synthetic_2(cfg: DictConfig) -> None:
    outliers_experiment(cfg)


if __name__ == "__main__":
    synthetic_2()
