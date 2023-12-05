import torch
import torch.nn.functional as F

from .base import BasePooling


class SegmentationConfidencePooling(BasePooling):
    def __init__(self):
        """Segmentation and confidence guided pooling.

        Filters all mobile classes from the feature map.
        """
        super().__init__("conf_segm")

        self.last_structure_class = 10

    def weighted_pooling(self, x: torch.Tensor, conf: torch.Tensor):
        x = x * conf
        x = torch.sum(x, dim=(2, 3)) / (torch.sum(conf, dim=(2, 3)) + 1e-8)

        return x  # 1, N, C

    def _forward(self, x: torch.Tensor, conf: torch.Tensor, segmentation: torch.Tensor):
        if x.dim() == 4:  # B, C, H, W
            mask = torch.ones(conf.shape).to(conf.device)
            mask[segmentation > self.last_structure_class] = 0.0

            conf = conf * mask

            return self.weighted_pooling(x, conf)
        else:
            raise ValueError(f"Expected input tensor to have 4 dimensions, got {x.dim()}")

    def forward(self, emb: torch.tensor, logits: torch.tensor, conf: torch.tensor):
        with torch.no_grad():
            logits_detached = logits.detach().clone()
            pred_detached = torch.argmax(logits_detached, dim=1).unsqueeze(1).type(torch.float32)  # B, 1, H, W

        shape_embedding = emb.shape[-2:]
        conf_resized = F.interpolate(conf, size=shape_embedding, mode="bilinear", align_corners=False)

        return self._forward(emb, conf_resized, pred_detached)
