import torch
import torch.nn as nn

from .base import BasePooling


class MyAdaptiveAvgPooling2D(BasePooling):
    def __init__(self, output_size: int = 1):
        """Custom adaptive average pooling layer."""
        super().__init__("average_pooling")

        if output_size != 1:
            raise NotImplementedError("output_size should be 1, otherwise the remaining method would not work.")

        self.output_size = output_size

    def _forward(self, x: torch.Tensor):
        if x.dim() == 4:
            x = nn.functional.adaptive_avg_pool2d(x, (self.output_size, self.output_size))
            x = x.flatten(start_dim=1)  # B, C, 1, 1 -> B, C
            return x
        else:
            raise ValueError(f"Expected input tensor to have 4 dimensions, got {x.dim()}")

    def forward(self, emb: torch.tensor, logits: torch.tensor, conf: torch.tensor):
        return self._forward(emb)
