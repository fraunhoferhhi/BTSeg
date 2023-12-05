import torch
import torch.nn as nn

from src.utils.logger_helper import add_member_variables_to_hparams_dict

from .backbones.mix_transformer import MixVisionTransformer
from .heads.segformer import SegFormerHead


class SegFormer(nn.Module):
    def __init__(
        self, num_classes: int, pretrained_weights_backbone: str = None, pretrained_weights_head: str = None
    ) -> None:
        super().__init__()

        if pretrained_weights_backbone is not None and pretrained_weights_backbone not in [
            "cityscapes",
        ]:
            raise ValueError(f"Unknown pretrained_backbone for segformer: {pretrained_weights_backbone}")

        if pretrained_weights_head is not None and pretrained_weights_head not in ["cityscapes"]:
            raise ValueError(f"Unknown pretrained_head for segformer: {pretrained_weights_head}")

        if num_classes <= 0:
            raise ValueError("The number of classes must be greater than 0.")

        self.num_classes = num_classes
        self.pretrained_weights_backbone = pretrained_weights_backbone
        self.pretrained_weights_head = pretrained_weights_head

        model_type = "mit_b5"

        self.backbone = MixVisionTransformer(model_type=model_type, pretrained=pretrained_weights_backbone)
        self.head = SegFormerHead(
            in_channels=self.backbone.arch_settings[model_type]["embed_dims"],  # [64, 128, 320, 512] for mit_b5
            in_index=[0, 1, 2, 3],
            channels=768,
            num_classes=self.num_classes,
            # input_transform="multiple_select",
            input_transform="resize_concat",
            pretrained=pretrained_weights_head,
        )

    def _forward(self, x):
        emb = self.backbone(x)

        with torch.cuda.amp.autocast(enabled=False):
            logits = self.head(emb)

        emb = self.head._transform_inputs(emb)

        return emb, logits

    def forward(self, x1, x2=None):
        emb_1, logits_1 = self._forward(x1)

        if x2 is not None:
            emb_2, logits_2 = self._forward(x2)

            return emb_1, emb_2, logits_1, logits_2

        return emb_1, logits_1

    @property
    def name(self):
        return "SegFormer"

    def save_hyperparameters(self, hparams_dict) -> None:
        add_member_variables_to_hparams_dict(
            hparams_dict,
            dict_params={
                "num_classes": self.num_classes,
            },
            key_prefix="SegFormer/",
        )
