import torch.nn as nn


class BasePooling(nn.Module):
    def __init__(self, name="base_pooling") -> None:
        super().__init__()

        self._name = name

    @property
    def name(self):
        return self._name
