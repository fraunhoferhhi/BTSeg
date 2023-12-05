# flake8: noqa: E402

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
from pathlib import Path

import largestinteriorrectangle as lir
import numpy as np
import torch
import torch.nn as nn

# import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from src.data.acdc import ACDCDataset
from src.data.data_transforms import Compose
from uawarpc.backbones.vgg import VGG
from uawarpc.heads.uawarpc import UAWarpCHead
from uawarpc.utils.warp import estimate_probability_of_confidence_interval, warp


def crop_image_with_rect(image, rect):
    # crop image to rect with opencv
    cropped_image = image[rect[0][1] : rect[1][1], rect[0][0] : rect[1][0]]

    return cropped_image


def get_intrinsic_rect(mask):
    rectangle = lir.lir(mask)

    return ((lir.pt1(rectangle)), (lir.pt2(rectangle)))


def align(alignment_backbone, alignment_head, images_ref, images_trg):
    assert alignment_backbone is not None
    assert alignment_head is not None

    h, w = images_ref.shape[-2:]

    images_trg_256 = nn.functional.interpolate(images_trg, size=(256, 256), mode="area")
    images_ref_256 = nn.functional.interpolate(images_ref, size=(256, 256), mode="area")

    x_backbone = alignment_backbone(torch.cat([images_ref, images_trg]), extract_only_indices=[-3, -2])
    unpacked_x = [torch.tensor_split(l, 2) for l in x_backbone]
    pyr_ref, pyr_trg = zip(*unpacked_x)
    x_backbone_256 = alignment_backbone(torch.cat([images_ref_256, images_trg_256]), extract_only_indices=[-2, -1])
    unpacked_x_256 = [torch.tensor_split(l, 2) for l in x_backbone_256]
    pyr_ref_256, pyr_trg_256 = zip(*unpacked_x_256)

    trg_ref_flow, trg_ref_uncert = alignment_head(pyr_trg, pyr_ref, pyr_trg_256, pyr_ref_256, (h, w))[-1]

    trg_ref_flow = nn.functional.interpolate(trg_ref_flow, size=(h, w), mode="bilinear", align_corners=False)
    trg_ref_uncert = nn.functional.interpolate(trg_ref_uncert, size=(h, w), mode="bilinear", align_corners=False)

    trg_ref_cert = estimate_probability_of_confidence_interval(trg_ref_uncert, R=1.5)

    warped_ref, trg_ref_mask = warp(images_ref, trg_ref_flow, return_mask=True)
    warp_confidence = trg_ref_mask.unsqueeze(1) * trg_ref_cert
    return warped_ref, warp_confidence


def create_warped_target_images(data_dir: Path):
    # fmt: off
    # exclude samples from the training set
    # superset of images smaller than 768x768 and images with 0 valid pixels/ patches according to 30_ACDC_statistics_uncertainty_patches.ipynb
    ids_samples_to_exclude = { "rain": [34, 41, 48, 48, 52, 55, 70, 101, 124, 164, 168, 169, 186, 190, 190, 191, 194, 194, 207, 308, 308, 316, 321, 322, 330, 349, 350, 350, 351, 351, 352, 353, 354, 375, 377, 380, 380, 381, 383, 384, 387, 388, 388, 389, 390, 391, 394],
                               "night": [0, 1, 10, 22, 33, 36, 36, 54, 56, 56, 57, 70, 72, 84, 89, 94, 126, 143, 145, 149, 159, 160, 180, 184, 189, 190, 190, 205, 206, 228, 249, 251, 252, 252, 261, 261, 300, 314, 319, 322, 330, 332, 336, 343, 352, 356, 360, 362, 365, 366, 373, 374, 378, 379, 381, 383, 384, 388, 395, 396, 399],
                               "fog": [0, 1, 2, 5, 6, 7, 8, 16, 24, 43, 44, 48, 49, 51, 52, 53, 54, 66, 78, 79, 80, 81, 82, 83, 84, 85, 86, 88, 89, 90, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 114, 115, 116, 117, 124, 125, 128, 129, 130, 132, 133, 141, 142, 143, 144, 147, 160, 176, 178, 181, 182, 188, 189, 190, 191, 192, 193, 231, 232, 233, 234, 236, 237, 244, 246, 247, 248, 255, 260, 263, 264, 265, 266, 267, 268, 269, 272, 299, 300, 304, 305, 306, 307, 308, 310, 311, 312, 313, 313, 315, 316, 317, 318, 319, 335, 337, 339, 340, 341, 342, 343, 344, 379, 384, 386, 392],
                               "snow": [0, 1, 2, 4, 17, 18, 19, 20, 46, 47, 48, 48, 101, 104, 104, 105, 105, 106, 107, 107, 109, 110, 111, 112, 113, 115, 120, 171, 179, 184, 185, 196, 199, 200, 207, 210, 216, 217, 221, 226, 227, 228, 231, 233, 248, 251, 253, 254, 255, 256, 257, 258, 259, 260, 261, 265, 271, 273, 279, 285, 290, 291, 292, 293, 295, 307, 308, 309, 311, 312, 313, 314, 315, 316, 342, 343, 344, 344, 345, 345, 346, 346, 347, 348, 355, 357, 358, 358, 359, 360, 372, 373, 376, 384, 384, 387]}
    # fmt: on

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alignment_backbone = VGG(model_type="vgg16", pretrained="imagenet", out_indices=[2, 3, 4]).to(device)
    alignment_head = UAWarpCHead(
        in_index=[0, 1], input_transform="multiple_select", estimate_uncertainty=True, pretrained="megadepth"
    ).to(device)

    alignment_backbone.eval()
    alignment_head.eval()

    conditions = ["rain", "fog", "snow", "night"]

    for condition in conditions:
        print(f"Condition: {condition}")

        ds = ACDCDataset(data_dir=data_dir, mode="train", weather_condition=condition, transform=Compose([]))

        len_ds = len(ds)

        for i in tqdm(range(len_ds)):
            data = ds[i]
            paths_sample = ds.get_paths_sample(i)

            with torch.no_grad():
                source, target = data["image"].unsqueeze(0).to(device), data["image_adv"].unsqueeze(0).to(device)
                warped_target, warp_confidence = align(alignment_backbone, alignment_head, target, source)

            warped_target = warped_target.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            warp_confidence = warp_confidence.squeeze().cpu().numpy()

            mask = warp_confidence > 0.0
            rect = get_intrinsic_rect(mask)

            w = rect[1][0] - rect[0][0]
            h = rect[1][1] - rect[0][1]
            if w < 768 or h < 768:
                if i not in ids_samples_to_exclude[condition]:
                    print("Rectangle is too small! But not already excluded.")

            # path to save warped target
            path_to_target = paths_sample["image_adv"]

            name_warped_target = path_to_target.stem + "_warped" + path_to_target.suffix
            name_uncertainty = path_to_target.stem + "_uncertainty" + ".npy"
            name_intrinsic_rect = path_to_target.stem + "_intrinsic_rect" + ".npy"

            path_output_target = path_to_target.parent

            file_output_warped_target = path_output_target / name_warped_target
            file_output_uncertainty = path_output_target / name_uncertainty
            file_output_intrinsic_rect = path_output_target / name_intrinsic_rect

            # save warped target
            im = Image.fromarray((warped_target * 255.0).astype("uint8"), "RGB")
            im.save(str(file_output_warped_target))

            # save uncertainty
            np.save(str(file_output_uncertainty), warp_confidence)

            # save intrinsic rectangle
            np.save(str(file_output_intrinsic_rect), rect)

            # print(f"Saved warped target to {file_output_warped_target}")


if __name__ == "__main__":
    # get environment variable DATA_DIR
    data_dir = Path(os.environ.get("DATA_DIR")) / "ACDC"

    create_warped_target_images(data_dir)
