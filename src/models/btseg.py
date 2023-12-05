import abc
from datetime import datetime
from functools import partial
from math import ceil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.image import imsave

# import wandb
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from torchvision.io import write_png

from src import utils
from src.utils.cityscape_utils import (
    get_colorized_image_from_predictions,
    get_label_train_classes,
)
from src.utils.logger_helper import add_member_variables_to_hparams_dict

# from torchinfo import summary


log = utils.get_pylogger(__name__)

torch.set_float32_matmul_precision("medium")


class OptimizerStrategy(abc.ABC):
    @abc.abstractmethod
    def get_optimizer_and_scheduler(self, trainer, module) -> Any:
        raise NotImplementedError("Implement this method in a subclass!")

    @abc.abstractmethod
    def save_hyperparameters(self, hparams_dict) -> None:
        raise NotImplementedError("Implement this method in a subclass!")


class BT_SemanticSegmentation(pl.LightningModule):
    """Implementation of a semantic segmentation model with a backbone pretrained using the Barlow
    Twins loss.

    Heavily inspired by: https://pytorch-
    lightning.readthedocs.io/en/stable/notebooks/lightning_examples/barlow-twins.html
    """

    def __init__(
        self,
        batch_size: int,
        effective_batch_size: int,
        optimizer_strategy: OptimizerStrategy,
        global_regularization_loss: nn.Module,
        segmentation_network: nn.Module,
        global_projection_head: nn.Module,
        pooling: nn.Module,
        segmentation_loss: nn.Module,
        balancing_factor_segmentation_bt: float,
        warm_up_steps_bt: int = 0,
        freeze_head: bool = False,
        num_classes_segmentation: int = 20,
        ignore_class_id: int = None,
        use_slide_inference: bool = True,
        height_slide_inference: int = 1080,
        width_slide_inference: int = 1080,
        predict_mode: str = None,
    ) -> None:
        """Implementation of a semantic segmentation model with a backbone pretrained using the
        Barlow Twins loss.

        Args:
            batch_size (int): The batch size.
            effective_batch_size (int): The effective batch size (num of vectors to store for calculation of the cross correlation matrix).
            optimizer_strategy (OptimizerStrategy): The optimizer strategy to use.
            global_regularization_loss (nn.Module): The regularization loss to use.
            segmentation_network (nn.Module): The segmentation network to use.
            global_projection_head (nn.Module): The projection head to use.
            pooling (nn.Module): The pooling layer to use.
            segmentation_loss (nn.Module): The segmentation loss to use.
            balancing_factor_segmentation_bt (float, optional): The factor by which the loss of the semantic segmentation task is multiplied.
            warm_up_steps_bt (int, optional): The number of steps for which the backbone is not trained from the projection head. Defaults to 0.
            freeze_head (bool, optional): Whether to freeze the head of the segmentation network. Defaults to False.
            loss_function (str, optional): The loss function to use for the semantic segmentation task. Defaults to "ce". Possible values: "ce", "dice", "logcosh_dice".
            num_classes_segmentation (int): The number of classes for the semantic segmentation task.
            ignore_class_id (int): The class id to ignore in the semantic segmentation task. Defaults to None.
            use_slide_inference (bool): Whether to use sliding window inference. Defaults to False.
            height_slide_inference (int): The height of the sliding window. Defaults to 1080.
            width_slide_inference (int): The width of the sliding window. Defaults to 1080.
            predict_mode (bool): Whether to output predictions as "matplotlib" files, "colored_png" (segmentation maps) or "png" for evaluation. Defaults to None.
        """
        super().__init__()
        optimizer_strategy.save_hyperparameters(self.hparams)
        global_regularization_loss.save_hyperparameters(self.hparams)
        segmentation_network.save_hyperparameters(self.hparams)
        global_projection_head.save_hyperparameters(self.hparams, "global_proj_")

        self.save_hyperparameters(
            ignore=[
                "optimizer_strategy",
                "global_regularization_loss",
                "segmentation_network",
                "global_projection_head",
                "pooling",
                "segmentation_loss",
            ]
        )

        if (
            isinstance(self.hparams.balancing_factor_segmentation_bt, str)
            and not self.hparams.balancing_factor_segmentation_bt == "auto"
        ):
            raise ValueError("balancing_factor_segmentation_bt must be a float or 'auto'")

        if self.hparams.balancing_factor_segmentation_bt == "auto":
            log.info("Using automatic balancing of the losses.")
        else:
            log.info(f"Using a loss balancing factor of {self.hparams.balancing_factor_segmentation_bt}.")

        assert (
            self.hparams.balancing_factor_segmentation_bt is not None
        ), "balancing_factor_segmentation_bt must be specified"

        self.optimizer_strategy = optimizer_strategy
        self.segmentation_network = segmentation_network
        self.pooling_1_1 = pooling
        self.loss_segmentation = segmentation_loss

        self.global_loss_bt = global_regularization_loss
        self.global_projection_head = global_projection_head

        if self.hparams.warm_up_steps_bt < effective_batch_size:
            log.warning(
                f"warm_up_steps_bt ({self.hparams.warm_up_steps_bt}) < effective_batch_size ({effective_batch_size}). Results in invalid batch norm statistics for the Barlow Twins loss. Setting warm_up_steps_bt to effective_batch_size."
            )
            self.hparams.warm_up_steps_bt = effective_batch_size

        # moving representation vector storage

        self.index_storage = 0
        self.index_storage_end = effective_batch_size

        # intermediate storage for the global representations
        size_buffer_global = (batch_size, effective_batch_size, self.global_projection_head.output_dim)

        self.register_buffer("global_representations_source", torch.zeros(size_buffer_global))
        self.register_buffer("global_representations_target", torch.zeros(size_buffer_global))

        if self.hparams.freeze_head:
            log.info("Freezing the head of the segmentation network.")
            for param in self.segmentation_network.head.parameters():
                param.requires_grad = False

        if self.hparams.use_slide_inference:
            log.info(
                f"Using sliding window inference. Window size: {self.hparams.height_slide_inference}x{self.hparams.width_slide_inference}."
            )

        # summary(self, input_size=(8, 3, 512, 512), col_names=("input_size", "output_size", "num_params"))

        # ====== Metrics ======

        joint_arguments = {
            "num_classes": self.hparams.num_classes_segmentation,
            "ignore_index": self.hparams.ignore_class_id,
        }

        # clean
        metric_collection_clean = MetricCollection(
            {
                "accuracy_macro_clean": MulticlassAccuracy(average="macro", **joint_arguments),
                "iou_macro_clean": MulticlassJaccardIndex(average="macro", **joint_arguments),
            }
        )
        # augmented
        metric_collection_adv = MetricCollection(
            {
                "accuracy_macro_adv": MulticlassAccuracy(average="macro", **joint_arguments),
                "iou_macro_adv": MulticlassJaccardIndex(average="macro", **joint_arguments),
            }
        )

        # Validation metrics
        self.val_metrics_clean = metric_collection_clean.clone(prefix="val/")
        self.val_metrics_adv = metric_collection_adv.clone(prefix="val/")

        # Test metrics
        self.test_metrics_clean = metric_collection_clean.clone(prefix="test/")
        self.test_metrics_adv = metric_collection_adv.clone(prefix="test/")

        self.iou_per_class_clean = MulticlassJaccardIndex(average="none", **joint_arguments)
        self.iou_per_class_adv = MulticlassJaccardIndex(average="none", **joint_arguments)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""

        input_shape = image.shape[-2:]

        if self.hparams.use_slide_inference:
            logits = self.slide_inference(image)
        else:
            logits = self.whole_inference(image)

        pred = torch.argmax(logits, dim=1)

        pred_shape = pred.shape[-2:]
        assert pred_shape == input_shape, f"Prediction shape {pred_shape} does not match input shape {input_shape}!"

        return pred

    def slide_inference(self, img):
        """
        ---------------------------------------------------------------------------
        Copyright (c) OpenMMLab. All rights reserved.

        This source code is licensed under the license found in the
        LICENSE file in https://github.com/open-mmlab/mmsegmentation.
        ---------------------------------------------------------------------------
        """
        batch_size, _, h_img, w_img = img.size()
        h_stride, w_stride, h_crop, w_crop = self.get_inference_slide_params(h_img, w_img)
        num_classes = self.segmentation_network.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.whole_inference(crop_img)
                preds += nn.functional.pad(
                    crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2))
                )
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        return preds  # attention: called preds, but actually logits

    def whole_inference(self, img):
        _, logits = self.segmentation_network(img)

        logits = F.interpolate(logits, size=img.shape[-2:], mode="bilinear", align_corners=False)

        return logits

    def training_step(self, batch, batch_idx):
        if batch_idx == 0 and not self.trainer.progress_bar_callback:
            log.info(f"Training epoch {self.trainer.current_epoch} started.")

        # forward pass

        x1 = batch["image"]
        x2 = batch["image_adv"]
        gt = batch["gt"]

        input_shape = x1.shape[-2:]

        emb_1, emb_2, logits_1, logits_2 = self.segmentation_network(x1, x2)

        # === Semantic Segmentation loss

        logits_1_upsampled = F.interpolate(logits_1, size=input_shape, mode="bilinear", align_corners=False)
        loss_segmentation = self.hparams.balancing_factor_segmentation_bt * self.loss_segmentation(
            logits_1_upsampled, gt
        )

        self.log("train/balancing_factor_segmentation_bt", self.hparams.balancing_factor_segmentation_bt)
        self.log("train/loss_ce", loss_segmentation, sync_dist=True)

        loss_total = loss_segmentation

        # === Barlow Twins loss

        if self.global_step < self.hparams.warm_up_steps_bt:
            if self.trainer.global_step == 0 and batch_idx == 0:
                log.info(
                    f"No backpropagation for the backbone from the projection heads for the first {self.hparams.warm_up_steps_bt} steps."
                )

            emb_1 = emb_1.detach()
            emb_2 = emb_2.detach()

        conf_alignment = batch["conf"].unsqueeze(1)

        f_1_global = self.pooling_1_1(emb_1, logits_1, conf_alignment)
        f_2_global = self.pooling_1_1(emb_2, logits_2, conf_alignment)

        z1_global = self.global_projection_head(f_1_global)  # B x D
        z2_global = self.global_projection_head(f_2_global)

        # moving representation vector storage
        self.global_representations_source[:, self.index_storage, :] = z1_global.detach().clone()
        self.global_representations_target[:, self.index_storage, :] = z2_global.detach().clone()

        # if self.global_step > self.hparams.warm_up_steps_bt:
        Z1_global = self.global_representations_source.clone()
        # B x B^ x N x D -> B^ (effective bs) -> number of samples used for the cross-correlation matrix
        Z1_global[:, self.index_storage, :] = z1_global
        Z2_global = self.global_representations_target.clone()
        Z2_global[:, self.index_storage, :] = z2_global

        loss_bt_global = self.global_loss_bt(Z1_global, Z2_global)["value"]
        self.log("train/loss_bt_global", loss_bt_global, sync_dist=True)

        loss_total += loss_bt_global

        # increment the index for the moving representation vector storage
        self.index_storage += 1
        if self.index_storage == self.index_storage_end:
            self.index_storage = 0

        # forward pass end
        self.log("train/loss", loss_total, sync_dist=True)

        # check if the loss is nan
        if torch.isnan(loss_total):
            raise ValueError("The loss is nan.")

        return loss_total

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0 and not self.trainer.sanity_checking and not self.trainer.progress_bar_callback:
            log.info(f"Validation epoch {self.trainer.current_epoch} started.")

        pred_clean = self(batch["image"])
        pred_adv = self(batch["image_adv"])

        self.val_metrics_clean.update(pred_clean, batch["gt"])
        self.val_metrics_adv.update(pred_adv, batch["gt_adv"])

    def on_validation_epoch_end(self, dataloader_idx=0):
        m = self.val_metrics_clean.compute()
        self.log_dict(m, sync_dist=True)
        self.val_metrics_clean.reset()

        m = self.val_metrics_adv.compute()
        self.log_dict(m, sync_dist=True)
        self.val_metrics_adv.reset()

    def test_step(self, batch, batch_idx):
        if batch_idx == 0 and torch.distributed.is_initialized():
            log.info(f"Test epoch {self.trainer.current_epoch} started.")

        # check if key image is in batch
        if "image" in batch:
            pred_clean = self(batch["image"])
            self.test_metrics_clean.update(pred_clean, batch["gt"])
            # compute IoU per class
            self.iou_per_class_clean.update(pred_clean, batch["gt"])

        if "image_adv" in batch:
            pred_adv = self(batch["image_adv"])
            self.test_metrics_adv.update(pred_adv, batch["gt_adv"])
            self.iou_per_class_adv.update(pred_adv, batch["gt_adv"])

    def on_test_epoch_end(self, dataloader_idx=0):
        m = self.test_metrics_clean.compute()
        self.log_dict(m, sync_dist=True)
        self.test_metrics_clean.reset()

        m = self.test_metrics_adv.compute()
        self.log_dict(m, sync_dist=True)
        self.test_metrics_adv.reset()

        # compute IoU per class
        columns = get_label_train_classes()[:-1]

        # check if wandb logger is used
        # flake8: noqa F821
        if "WandbLogger" in self.trainer.logger.__class__.__name__:
            data_clean = [self.iou_per_class_clean.compute().tolist()]
            table_iou_per_class_clean = wandb.Table(data=data_clean, columns=columns)
            self.trainer.logger.experiment.log({"IoU per class clean": table_iou_per_class_clean})

            data_adv = [self.iou_per_class_adv.compute().tolist()]
            table_iou_per_class_adv = wandb.Table(data=data_adv, columns=columns)
            self.trainer.logger.experiment.log({"IoU per class adv": table_iou_per_class_adv})

        self.iou_per_class_clean.reset()
        self.iou_per_class_adv.reset()

    def on_predict_start(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.output_folder = Path(f"./output/Pred_BT_SemanticSegmentation/{timestamp}")
        self.output_folder.mkdir(parents=True, exist_ok=False)

        log.info(f"Writing predictions into {self.output_folder}...")

    def predict_step(self, batch: Any, batch_idx: int) -> None:
        pred_clean = self(batch["image"])
        pred_adv = self(batch["image_adv"])

        pred_dataset = self.trainer.datamodule.predict_dataloader().dataset

        batch_size = batch["image"].size(0)

        for i in range(batch_size):
            if self.hparams.predict_mode == "matplotlib":
                self.output_matplotlib_plot(batch, batch_idx, pred_clean, pred_adv, batch_size, i, pred_dataset)
            elif self.hparams.predict_mode == "png":
                self.output_png_eval(batch_idx, pred_clean, pred_adv, pred_dataset, batch_size, i)
            elif self.hparams.predict_mode == "colored_png":
                self.output_colored_png(batch_idx, pred_clean, pred_adv, pred_dataset, batch_size, i)
            else:
                raise ValueError(f"Unknown predict mode {self.hparams.predict_mode}.")

    def get_inference_slide_params(self, h, w):
        def get_para(dim_inf, dim_train):
            h_blocks = ceil(dim_inf / dim_train)

            if h_blocks > 1:
                num_spaces = h_blocks - 1

                overlap_all = int(dim_train * h_blocks - dim_inf)

                # residual = overlap_all % num_spaces

                overlap_each = overlap_all // num_spaces

                stride = dim_train - overlap_each
                return stride, dim_train

            else:
                return 1, dim_inf

        h_stride, h_crop = get_para(h, self.hparams.height_slide_inference)
        w_stride, w_crop = get_para(w, self.hparams.width_slide_inference)

        return h_stride, w_stride, h_crop, w_crop

    def output_colored_png(self, batch_idx, pred_clean, pred_adv, pred_dataset, batch_size, i):
        data_id = batch_idx * batch_size + i
        paths_data = pred_dataset.get_paths_sample(data_id)

        pred_colored = get_colorized_image_from_predictions(pred_adv[i].cpu().numpy())

        imsave(str(self.output_folder / paths_data["image_adv"].stem) + "_pred.png", pred_colored)

    def output_png_eval(self, batch_idx, pred_clean, pred_adv, pred_dataset, batch_size, i):
        data_id = batch_idx * batch_size + i
        paths_data = pred_dataset.get_paths_sample(data_id)

        # write_png(
        #     pred_clean[i].unsqueeze(0).cpu().type(torch.uint8),
        #     filename=str(self.output_folder / paths_data["image"].name),
        #     compression_level=6,
        # )
        write_png(
            pred_adv[i].unsqueeze(0).cpu().type(torch.uint8),
            filename=str(self.output_folder / paths_data["image_adv"].name),
            compression_level=6,
        )

    def output_matplotlib_plot(self, batch, batch_idx, pred_clean, pred_adv, batch_size, i, pred_dataset):
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        data_id = batch_idx * batch_size + i  # assumes unshuffled dataset!
        paths_data = pred_dataset.get_paths_sample(data_id)

        name_file = paths_data["image"].stem
        weather_condition = paths_data["image"].parent.parent.parent.name

        image_np = np.clip((std * batch["image"][i].cpu().numpy() + mean).transpose(1, 2, 0), 0.0, 1.0)
        image_adv_np = np.clip((std * batch["image_adv"][i].cpu().numpy() + mean).transpose(1, 2, 0), 0.0, 1.0)

        gt = get_colorized_image_from_predictions(batch["gt"][i].cpu().numpy())
        gt_adv = get_colorized_image_from_predictions(batch["gt_adv"][i].cpu().numpy())

        pred_clean_np = get_colorized_image_from_predictions(pred_clean[i].cpu().numpy())
        pred_adv_np = get_colorized_image_from_predictions(pred_adv[i].cpu().numpy())

        fig, ax = plt.subplots(2, 3, figsize=(15, 10))

        ax[0, 0].imshow(image_np)
        ax[0, 0].set_title("Image Reference")
        ax[0, 0].axis("off")

        ax[0, 1].imshow(gt)
        ax[0, 1].set_title("GT Reference")
        ax[0, 1].axis("off")

        ax[0, 2].imshow(pred_clean_np)
        ax[0, 2].set_title("Pred Reference")
        ax[0, 2].axis("off")

        ax[1, 0].imshow(image_adv_np)
        ax[1, 0].set_title(f"Image Adverse")
        ax[1, 0].axis("off")

        ax[1, 1].imshow(gt_adv)
        ax[1, 1].set_title(f"GT Adverse")
        ax[1, 1].axis("off")

        ax[1, 2].imshow(pred_adv_np)
        ax[1, 2].set_title(f"Pred Adverse")
        ax[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig(str(self.output_folder) + f"/pred_{name_file}_{weather_condition}.png")
        plt.close()

    def on_predict_end(self) -> None:
        log.info("Prediction finished!")

    def configure_optimizers(self):
        return self.optimizer_strategy.get_optimizer_and_scheduler(self.trainer, self)


class LRWarmupAndPolyDecay(OptimizerStrategy):
    def __init__(
        self,
        lr_classifier: float = 0.00001,
        lr_backbone: float = None,
        lr_projection_head_global: float = None,
        warmup_steps: int = 1500,
        exp_poly: float = 0.9,
    ):
        """Strategy with AdamW optimizer and learning rate warm-up and poly decay.

        Args:
            lr_classifier (float, optional): Learning rate for the classifier. Defaults to 0.00001.
            lr_backbone (float, optional): Learning rate for the backbone. Defaults to lr_classifier.
            lr_projection_head_global (float, optional): Learning rate for the global projection head. Defaults to lr_classifier.
            warmup_steps: Number of steps for the warm-up. Default: 1500
            exp_poly: Exponent for the polynomial decay. Default: 0.9
        """

        self.lr_classifier = lr_classifier
        self.lr_backbone = lr_backbone if lr_backbone is not None else lr_classifier
        self.lr_projection_head_global = (
            lr_projection_head_global if lr_projection_head_global is not None else lr_classifier
        )

        self.warmup_steps = warmup_steps
        self.exp_poly = exp_poly

    def save_hyperparameters(self, hparams_dict) -> None:
        add_member_variables_to_hparams_dict(
            hparams_dict, self.__dict__, "opt_", {"strategy": "bt_warm-up+sh_poly-decay"}
        )

    def get_optimizer_and_scheduler(self, trainer, module):
        max_number_of_steps = trainer.max_steps
        log.info(f"Max. number of steps: {max_number_of_steps}")

        assert self.warmup_steps < max_number_of_steps, "warmup_steps >= max_steps"

        lr_decay_steps = max_number_of_steps - self.warmup_steps

        # ======= CREATE OPTIMIZER =======
        params_backbone = list(module.segmentation_network.backbone.parameters())
        params_head = [p for p in module.segmentation_network.head.parameters() if p.requires_grad]

        params_projection_head_global = (
            list(module.global_projection_head.parameters()) if module.global_projection_head else []
        )

        assert not len(list(module.parameters())) == 0, "No parameters found."
        assert len(params_backbone) + len(params_head) + len(params_projection_head_global) == len(
            list(module.parameters())
        ), "Not all parameters are assigned to a group."

        params = [
            {
                "params": params_backbone,
                "lr": self.lr_backbone,
                "name": "backbone",
            },
            {
                "params": params_head,
                "lr": self.lr_classifier,
                "name": "classifier",
            },
            {
                "params": params_projection_head_global,
                "lr": self.lr_projection_head_global,
                "name": "projection_head_global",
            },
        ]

        opt = torch.optim.AdamW(params, weight_decay=0.01)

        # ======= CREATE LR SCHEDULER =======

        log.info("Using lr scheduler!")
        log.info(f"lr_warmup_steps: {self.warmup_steps}")
        log.info(f"lr_decay_steps: {lr_decay_steps}")

        def warmup_and_poly_decay_func(warmup_steps, decay_steps, exp_poly, step):
            if step <= warmup_steps:
                factor = min(step / warmup_steps, 1)
            else:
                factor = (1.0 - (step - warmup_steps) / decay_steps) ** exp_poly
            return factor

        lambda_func = []
        for _ in range(len(params)):
            lambda_func.append(partial(warmup_and_poly_decay_func, self.warmup_steps, lr_decay_steps, self.exp_poly))

        scheduler = LambdaLR(
            opt,
            lr_lambda=lambda_func,
        )

        # return opt and scheduler as dict
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
