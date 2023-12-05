import numpy as np

# import seaborn as sns
import torch
import wandb
from pytorch_lightning import Trainer

from src import utils
from src.utils.cityscape_utils import get_train_id_to_name_mapping

log = utils.get_pylogger(__name__)


def wandb_add_image_figure(trainer: Trainer, tag: str, image: torch.Tensor) -> None:
    """Add image to wandb.

    Args:
        trainer (Trainer): Trainer object.
        tag (str): Tag for the figure.
        image (torch.Tensor): Image tensor. Gets denormalized with mean and std.
    """

    image = image.cpu().numpy()

    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0]
        elif image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        else:
            raise ValueError(f"Image has wrong shape {image.shape}")
    elif image.ndim == 2:
        pass
    else:
        raise ValueError(f"Image has wrong shape {image.shape}")

    wandb_image = wandb.Image(
        image,
    )

    trainer.logger.experiment.log({tag: wandb_image})

    log.debug(f"Logged image to wandb with tag {tag} and global step {trainer.global_step}.")


def wandb_add_cityscapes_image_gt_figure(
    trainer: Trainer, tag: str, images: list[torch.Tensor], gts: list[torch.Tensor]
) -> None:
    """Add cityscapes image and ground truth to wandb.

    Args:
        trainer (Trainer): Trainer object.
        tag (str): Tag for the figure.
    """

    # create numpy array from mean
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    masked_images = []

    for image, gt in zip(images, gts):
        image_np = np.clip((std * image.cpu().numpy() + mean).transpose(1, 2, 0), 0.0, 1.0)
        gt = gt.cpu().numpy()

        traind_id_to_name_mapping = get_train_id_to_name_mapping()

        # unlabbled is 255 in gt
        gt[gt == 19] = 255

        masked_image = wandb.Image(
            image_np,
            masks={
                "ground truth": {"mask_data": gt, "class_labels": traind_id_to_name_mapping},
            },
        )

        masked_images.append(masked_image)

    trainer.logger.experiment.log({tag: masked_images})

    log.debug(f"Logged image to wandb with tag {tag} and global step {trainer.global_step}.")


def wandb_add_cityscapes_image_pred_gt_figure(
    trainer: Trainer, tag: str, image: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor
) -> None:
    """Add cityscapes image, prediction and ground truth to wandb.

    Args:
        trainer (Trainer): Trainer object.
        tag (str): Tag for the figure.
        image (torch.Tensor): Image tensor. Gets denormalized with mean and std.
        pred (torch.Tensor): Prediction tensor. Should contain class ids not probabilities.
        gt (torch.Tensor): Ground truth tensor.
    """

    # create numpy array from mean
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    image_np = np.clip((std * image.cpu().numpy() + mean).transpose(1, 2, 0), 0.0, 1.0)
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()

    traind_id_to_name_mapping = get_train_id_to_name_mapping()

    # unlabbled is 255 in gt
    gt[gt == 19] = 255
    pred[pred == 19] = 255

    masked_image = wandb.Image(
        image_np,
        masks={
            "predictions": {"mask_data": pred, "class_labels": traind_id_to_name_mapping},
            "ground truth": {"mask_data": gt, "class_labels": traind_id_to_name_mapping},
        },
    )

    trainer.logger.experiment.log({tag: masked_image})

    log.debug(f"Logged image to wandb with tag {tag} and global step {trainer.global_step}.")


def wandb_add_cityscapes_image_weights_figure(
    trainer: Trainer, tag: str, image: torch.Tensor, weights: torch.Tensor
) -> None:
    """Add cityscapes image, prediction and ground truth to wandb.

    Args:
        trainer (Trainer): Trainer object.
        tag (str): Tag for the figure.
        image (torch.Tensor): Image tensor. Gets denormalized with mean and std.
        weights (torch.Tensor, optional): Weights tensor. Defaults to None.
    """

    # create numpy array from mean
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    image_np = np.clip((std * image.cpu().numpy() + mean).transpose(1, 2, 0), 0.0, 1.0)

    weights = np.clip(weights.cpu().numpy() * 255.0, 0.0, 255.0).astype(np.uint8)

    wandb_image = wandb.Image(
        image_np,
        caption="Image",
    )

    wandb_weights = wandb.Image(
        weights,
        caption="Weights",
    )

    trainer.logger.experiment.log({tag: [wandb_image, wandb_weights]})

    log.debug(f"Logged image and weights to wandb with tag {tag} and global step {trainer.global_step}.")
