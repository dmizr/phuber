import logging
import re

from omegaconf import DictConfig, OmegaConf


def to_clean_str(s: str) -> str:
    """Keeps only alphanumeric characters and lowers them

    Args:
        s: a string

    Returns:
        cleaned string
    """
    return re.sub("[^a-zA-Z0-9]", "", s).lower()


def display_config(cfg: DictConfig):
    logger = logging.getLogger()
    logger.info("Configuration:\n")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 40 + "\n")


def flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """ Recursively flattens a dictionary

    Args:
        d: Dictionary to flatten
        parent_key: key of parent dictionary
        sep: separator between key and child key

    Returns:
        flattened dictionary

    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
