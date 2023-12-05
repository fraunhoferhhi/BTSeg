import torch
import torch.nn.functional as F

from .base import BasePooling


class ConfWeightedPooling(BasePooling):
    def __init__(self):
        """Pooling guided by the confidence of the warping."""
        super().__init__("confidence_weighted_pooling")

    def weighted_pooling(self, x: torch.Tensor, conf: torch.Tensor):
        x = x * conf
        x = torch.sum(x, dim=(2, 3)) / (torch.sum(conf, dim=(2, 3)) + 1e-8)

        return x  # 1, N, C

    def _forward(self, x: torch.Tensor, conf: torch.Tensor):
        if x.dim() == 4:  # B, C, H, W
            return self.weighted_pooling(x, conf)
        else:
            raise ValueError(f"Expected input tensor to have 4 dimensions, got {x.dim()}")

    def forward(self, emb: torch.tensor, logits: torch.tensor, conf: torch.tensor):
        shape_embedding = emb.shape[-2:]
        conf_resized = F.interpolate(conf, size=shape_embedding, mode="bilinear", align_corners=False)

        return self._forward(emb, conf_resized)
