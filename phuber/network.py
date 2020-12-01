import torch
import torch.nn as nn


class LeNet(nn.Module):
    """LeNet-5 from `"Gradient-Based Learning Applied To Document Recognition"
    <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>`_
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.tanh = nn.Tanh()
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(self.tanh(self.conv1(x)))
        x = self.avgpool(self.tanh(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
