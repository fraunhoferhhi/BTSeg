import torch

from .base import BasePooling


class SegmentationGuidedPooling(BasePooling):
    def __init__(self):
        """Segmentation guided pooling.

        Filters all mobile classes from the feature map.
        """
        super().__init__("segm_pooling")

        self.last_structure_class_cityscapes = 10

    def seg_pooling(self, x: torch.Tensor, segmentation: torch.Tensor):
        filter_mobile_classes = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])).to(x.device)  # bx1xhxw
        filter_mobile_classes[segmentation > self.last_structure_class_cityscapes] = 0.0

        num_non_mobile = filter_mobile_classes.sum(dim=(2, 3))

        x = (x * filter_mobile_classes).sum(dim=(2, 3)) / (num_non_mobile + 1e-8)  # bxk

        return x, filter_mobile_classes

    def _forward(self, x: torch.Tensor, segmentation: torch.Tensor):
        # check segmentation is detached
        if segmentation.requires_grad:
            raise ValueError("Expected segmentation to be detached")

        if x.dim() == 4:
            if segmentation.dim() == 3:
                segmentation = segmentation.unsqueeze(1)  # bx1xhxw

            x, filter_mobile_classes = self.seg_pooling(x, segmentation)

            return {"features": x, "weights": filter_mobile_classes}
        else:
            raise ValueError(f"Expected input tensor to have 4 dimensions, got {x.dim()}")

    def forward(self, emb: torch.tensor, logits: torch.tensor, conf: torch.tensor):
        with torch.no_grad():
            logits_detached = logits.detach().clone()
            pred_detached = torch.argmax(logits_detached, dim=1).unsqueeze(1).type(torch.float32)  # B, 1, H, W

        return self._forward(emb, pred_detached)["features"]
