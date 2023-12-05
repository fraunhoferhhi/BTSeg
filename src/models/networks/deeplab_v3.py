import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabV3(nn.Module):
    def __init__(
        self, num_classes: int, pretrained_weights_backbone: str = None, pretrained_weights_head: str = None
    ) -> None:
        super().__init__()

        if pretrained_weights_backbone is not None and pretrained_weights_backbone not in ["imagenet"]:
            raise NotImplementedError("Only imagenet pretrained weights for the backbone are supported at the moment.")

        if pretrained_weights_head is not None:
            raise NotImplementedError("Pretrained weights for the head are not supported at the moment.")

        if num_classes <= 0:
            raise ValueError("The number of classes must be greater than 0.")

        if pretrained_weights_backbone == "imagenet":
            weights_backbone = ResNet50_Weights.IMAGENET1K_V1

        self.num_classes = num_classes
        self.pretrained_weights_backbone = pretrained_weights_backbone
        self.pretrained_weights_head = pretrained_weights_head

        deeplab_model = deeplabv3_resnet50(
            num_classes=self.num_classes,
            aux_loss=False,
            weights_backbone=weights_backbone,
        )

        # set bn momentum to 0.01 for backbone
        for m in deeplab_model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.01

        self.backbone = deeplab_model.backbone
        self.head = deeplab_model.classifier

        self.output_dim = self.head[4].out_channels

    def _forward(self, x):
        emb = self.backbone(x)["out"]
        logits = self.head(emb)

        return emb, logits

    def forward(self, x1, x2=None):
        emb_1, logits_1 = self._forward(x1)

        if x2 is not None:
            emb_2, logits_2 = self._forward(x2)

            return emb_1, emb_2, logits_1, logits_2

        return emb_1, logits_1

    @property
    def name(self):
        return "DeepLabV3"

    def save_hyperparameters(self, hparams_dict) -> None:
        hparams_dict["num_classes"] = self.num_classes
        hparams_dict["pretrained_weights_backbone"] = self.pretrained_weights_backbone
        hparams_dict["pretrained_weights_head"] = self.pretrained_weights_head
