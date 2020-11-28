import logging
from typing import Optional

import torch
import tqdm
from torch.utils.data import DataLoader

from phuber.metrics import AccuracyMetric


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        loader: DataLoader,
        checkpoint_path: Optional[str] = None,
    ) -> None:

        # Logging
        self.logger = logging.getLogger()

        # Device
        self.device = device

        # Data
        self.loader = loader

        # Model
        self.model = model

        if checkpoint_path:
            self._load_from_checkpoint(checkpoint_path)

        # Metrics
        self.acc_metric = AccuracyMetric(k=1)

    def evaluate(self) -> float:
        """ Evaluates the model

        Returns:
            (float) accuracy (on a 0 to 1 scale)

        """

        # Progress bar
        pbar = tqdm.tqdm(total=len(self.loader), leave=False)
        pbar.set_description("Evaluating... ")

        # Set to eval
        self.model.eval()

        # Loop
        for data, target in self.loader:
            with torch.no_grad():
                # To device
                data, target = data.to(self.device), target.to(self.device)

                # Forward
                out = self.model(data)

                self.acc_metric.update(out, target)

                # Update progress bar
                pbar.update()

        pbar.close()

        accuracy = self.acc_metric.compute()
        self.logger.info(f"Accuracy: {accuracy:.4f}\n")

        return accuracy

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
