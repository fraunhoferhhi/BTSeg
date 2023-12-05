import json
import os
from collections import namedtuple
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src import utils
from src.data.data_transforms import Compose, Normalize, ToTensor

log = utils.get_pylogger(__name__)


class ACGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        weather_condition: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        """Creates a PyTorch Lightning Data Module for the ACG bencmark.

        Paper: https://arxiv.org/abs/2303.05194
        Source: https://data.vision.ee.ethz.ch/brdavid/cma/ACG.zip

        Args:
            data_dir (Path): Path to the ACG dataset.
            weather_condition (str): Weather condition to use. ["all", "rain", "snow", "fog", "night"]
            batch_size (int): Batch size for the dataloader.
            num_workers (int): Number of workers for the dataloader.
            pin_memory (bool, optional): Pin memory for the dataloader. Defaults to True.
            persistent_workers (bool, optional): Persistent workers for the dataloader. Defaults to True.
        """

        super().__init__()
        self.save_hyperparameters(logger=True)

        if isinstance(self.hparams.data_dir, str):
            self.hparams.data_dir = Path(self.hparams.data_dir)

        self.hparams.data_dir = self.hparams.data_dir / "ACG_benchmark"

        self.transform = Compose(
            [
                ToTensor(),
                Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # needed for application of ResNet, VGGNet, ...
            ]
        )

        if self.hparams.weather_condition not in ["all", "rain", "snow", "fog", "night"]:
            raise ValueError(f"Invalid weather condition: {self.hparams.weather_condition}")

        if self.hparams.weather_condition == "all":
            self.hparams.weather_condition = ["rain", "snow", "fog", "night"]

        if isinstance(self.hparams.weather_condition, str):
            self.hparams.weather_condition = [self.hparams.weather_condition]

    def prepare_data(self):
        """Prepares the data for the usage in a dataset.

        If needed, the download of image data could happen here. The function will not be
        distributed -> do not assign a state here! (self.x = y)
        """
        pass

    def setup(self, stage: str = None):
        """Loads the appropriate ACDC datasets depending on the stage parameter.

        Args:
            stage (str, optional): Set the network stage. ["fit", "test" or "predict"]. Defaults to None.

        Raises:
            RuntimeError: For unknown stage parameter options.
        """

        if stage == "fit":
            raise RuntimeError(
                "Stage 'fit' is not supported for the ACG dataset! Only 'test' and 'predict' are supported."
            )

        self.acg_test = ACG(
            self.hparams.data_dir,
            self.hparams.weather_condition,
            transforms=self.transform,
        )

    def train_dataloader(self):
        raise RuntimeError("Stage 'fit' is not supported for the ACG dataset! Only 'test' and 'predict' are supported.")

    def val_dataloader(self):
        raise RuntimeError("Stage 'fit' is not supported for the ACG dataset! Only 'test' and 'predict' are supported.")

    def test_dataloader(self):
        return DataLoader(
            self.acg_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.acdc_predict,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )


# below copy from https://data.vision.ee.ethz.ch/brdavid/cma/ACG.zip -> ACG.py


class ACG(torch.utils.data.Dataset):
    """ACG benchmark from the paper:

    Contrastive Model Adaptation for Cross-Condition Robustness in Semantic Segmentation
    """

    WildDash2Class = namedtuple("WildDash2Class", ["name", "id", "train_id"])
    labels = [
        #       name                             id    trainId
        WildDash2Class("unlabeled", 0, 255),
        WildDash2Class("ego vehicle", 1, 255),
        WildDash2Class("overlay", 2, 255),
        WildDash2Class("out of roi", 3, 255),
        WildDash2Class("static", 4, 255),
        WildDash2Class("dynamic", 5, 255),
        WildDash2Class("ground", 6, 255),
        WildDash2Class("road", 7, 0),
        WildDash2Class("sidewalk", 8, 1),
        WildDash2Class("parking", 9, 255),
        WildDash2Class("rail track", 10, 255),
        WildDash2Class("building", 11, 2),
        WildDash2Class("wall", 12, 3),
        WildDash2Class("fence", 13, 4),
        WildDash2Class("guard rail", 14, 255),
        WildDash2Class("bridge", 15, 255),
        WildDash2Class("tunnel", 16, 255),
        WildDash2Class("pole", 17, 5),
        WildDash2Class("polegroup", 18, 255),
        WildDash2Class("traffic light", 19, 6),
        WildDash2Class("traffic sign front", 20, 7),
        WildDash2Class("vegetation", 21, 8),
        WildDash2Class("terrain", 22, 9),
        WildDash2Class("sky", 23, 10),
        WildDash2Class("person", 24, 11),
        WildDash2Class("rider", 25, 12),
        WildDash2Class("car", 26, 13),
        WildDash2Class("truck", 27, 14),
        WildDash2Class("bus", 28, 15),
        WildDash2Class("caravan", 29, 255),
        WildDash2Class("trailer", 30, 255),
        WildDash2Class("on rails", 31, 16),
        WildDash2Class("motorcycle", 32, 17),
        WildDash2Class("bicycle", 33, 18),
        WildDash2Class("pickup", 34, 14),
        WildDash2Class("van", 35, 13),
        WildDash2Class("billboard", 36, 255),
        WildDash2Class("street light", 37, 255),
        WildDash2Class("road marking", 38, 0),
        WildDash2Class("junctionbox", 39, 255),
        WildDash2Class("mailbox", 40, 255),
        WildDash2Class("manhole", 41, 0),
        WildDash2Class("phonebooth", 42, 255),
        WildDash2Class("pothole", 43, 0),
        WildDash2Class("bikerack", 44, 255),
        WildDash2Class("traffic sign frame", 45, 5),
        WildDash2Class("utility pole", 46, 5),
        WildDash2Class("motorcyclist", 47, 12),
        WildDash2Class("bicyclist", 48, 12),
        WildDash2Class("other rider", 49, 12),
        WildDash2Class("bird", 50, 255),
        WildDash2Class("ground animal", 51, 255),
        WildDash2Class("curb", 52, 1),
        WildDash2Class("traffic sign any", 53, 255),
        WildDash2Class("traffic sign back", 54, 255),
        WildDash2Class("trashcan", 55, 255),
        WildDash2Class("other barrier", 56, 3),
        WildDash2Class("other vehicle", 57, 255),
        WildDash2Class("auto rickshaw", 58, 17),
        WildDash2Class("bench", 59, 255),
        WildDash2Class("mountain", 60, 255),
        WildDash2Class("tram track", 61, 0),
        WildDash2Class("wheeled slow", 62, 255),
        WildDash2Class("boat", 63, 255),
        WildDash2Class("bikelane", 64, 0),
        WildDash2Class("bikelane sidewalk", 65, 1),
        WildDash2Class("banner", 66, 255),
        WildDash2Class("dashcam mount", 67, 255),
        WildDash2Class("water", 68, 255),
        WildDash2Class("sand", 69, 255),
        WildDash2Class("pedestrian area", 70, 0),
        WildDash2Class("fire hydrant", 71, 255),
        WildDash2Class("cctv camera", 72, 255),
        WildDash2Class("snow", 73, 255),
        WildDash2Class("catch basin", 74, 0),
        WildDash2Class("crosswalk plain", 75, 0),
        WildDash2Class("crosswalk zebra", 76, 0),
        WildDash2Class("manhole sidewalk", 77, 1),
        WildDash2Class("curb terrain", 78, 9),
        WildDash2Class("service lane", 79, 0),
        WildDash2Class("curb cut", 80, 1),
        WildDash2Class("license plate", -1, 255),
    ]
    id_to_train_id = np.array([c.train_id for c in labels], dtype=int)

    def __init__(
        self,
        root: str,
        conditions: Union[List[str], str] = ["fog", "night", "rain", "snow"],
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.conditions = [conditions] if isinstance(conditions, str) else conditions
        self.transforms = transforms

        # load WildDash2 encodings
        pan = json.load(open(os.path.join(self.root, "WildDash2", "panoptic.json")))
        self.img_to_segments = {img_dict["file_name"]: img_dict["segments_info"] for img_dict in pan["annotations"]}

        self.paths = {k: [] for k in ["image", "semantic", "dataset"]}
        paths_night = {k: [] for k in ["image", "semantic", "dataset"]}
        paths_rain = {k: [] for k in ["image", "semantic", "dataset"]}
        paths_snow = {k: [] for k in ["image", "semantic", "dataset"]}
        paths_fog = {k: [] for k in ["image", "semantic", "dataset"]}

        wilddash_img_ids_fog = [i_id.strip() for i_id in open(os.path.join(self.root, "ACG", "ACG_WildDash_fog"))]
        for file_name in wilddash_img_ids_fog:
            file_path = os.path.join(self.root, "WildDash2", file_name)
            semantic_name = file_name.replace("images/", "panoptic/").replace(".jpg", ".png")
            semantic_path = os.path.join(self.root, "WildDash2", semantic_name)
            paths_fog["image"].append(file_path)
            paths_fog["semantic"].append(semantic_path)
            paths_fog["dataset"].append("wilddash")

        wilddash_img_ids_night = [i_id.strip() for i_id in open(os.path.join(self.root, "ACG", "ACG_WildDash_night"))]
        for file_name in wilddash_img_ids_night:
            file_path = os.path.join(self.root, "WildDash2", file_name)
            semantic_name = file_name.replace("images/", "panoptic/").replace(".jpg", ".png")
            semantic_path = os.path.join(self.root, "WildDash2", semantic_name)
            paths_night["image"].append(file_path)
            paths_night["semantic"].append(semantic_path)
            paths_night["dataset"].append("wilddash")

        wilddash_img_ids_rain = [i_id.strip() for i_id in open(os.path.join(self.root, "ACG", "ACG_WildDash_rain"))]
        for file_name in wilddash_img_ids_rain:
            file_path = os.path.join(self.root, "WildDash2", file_name)
            semantic_name = file_name.replace("images/", "panoptic/").replace(".jpg", ".png")
            semantic_path = os.path.join(self.root, "WildDash2", semantic_name)
            paths_rain["image"].append(file_path)
            paths_rain["semantic"].append(semantic_path)
            paths_rain["dataset"].append("wilddash")

        wilddash_img_ids_snow = [i_id.strip() for i_id in open(os.path.join(self.root, "ACG", "ACG_WildDash_snow"))]
        for file_name in wilddash_img_ids_snow:
            file_path = os.path.join(self.root, "WildDash2", file_name)
            semantic_name = file_name.replace("images/", "panoptic/").replace(".jpg", ".png")
            semantic_path = os.path.join(self.root, "WildDash2", semantic_name)
            paths_snow["image"].append(file_path)
            paths_snow["semantic"].append(semantic_path)
            paths_snow["dataset"].append("wilddash")

        bdd100k_img_ids_fog = [i_id.strip() for i_id in open(os.path.join(self.root, "ACG", "ACG_BDD100k_fog"))]
        for file_name in bdd100k_img_ids_fog:
            file_path = os.path.join(self.root, "bdd100k", file_name)
            semantic_name = file_name.replace("images/10k/", "labels/sem_seg/masks/").replace(".jpg", ".png")
            semantic_path = os.path.join(self.root, "bdd100k", semantic_name)
            paths_fog["image"].append(file_path)
            paths_fog["semantic"].append(semantic_path)
            paths_fog["dataset"].append("bdd100k")

        bdd100k_img_ids_night = [i_id.strip() for i_id in open(os.path.join(self.root, "ACG", "ACG_BDD100k_night"))]
        for file_name in bdd100k_img_ids_night:
            file_path = os.path.join(self.root, "bdd100k", file_name)
            semantic_name = file_name.replace("images/10k/", "labels/sem_seg/masks/").replace(".jpg", ".png")
            semantic_path = os.path.join(self.root, "bdd100k", semantic_name)
            paths_night["image"].append(file_path)
            paths_night["semantic"].append(semantic_path)
            paths_night["dataset"].append("bdd100k")

        bdd100k_img_ids_rain = [i_id.strip() for i_id in open(os.path.join(self.root, "ACG", "ACG_BDD100k_rain"))]
        for file_name in bdd100k_img_ids_rain:
            file_path = os.path.join(self.root, "bdd100k", file_name)
            semantic_name = file_name.replace("images/10k/", "labels/sem_seg/masks/").replace(".jpg", ".png")
            semantic_path = os.path.join(self.root, "bdd100k", semantic_name)
            paths_rain["image"].append(file_path)
            paths_rain["semantic"].append(semantic_path)
            paths_rain["dataset"].append("bdd100k")

        bdd100k_img_ids_snow = [i_id.strip() for i_id in open(os.path.join(self.root, "ACG", "ACG_BDD100k_snow"))]
        for file_name in bdd100k_img_ids_snow:
            file_path = os.path.join(self.root, "bdd100k", file_name)
            semantic_name = file_name.replace("images/10k/", "labels/sem_seg/masks/").replace(".jpg", ".png")
            semantic_path = os.path.join(self.root, "bdd100k", semantic_name)
            paths_snow["image"].append(file_path)
            paths_snow["semantic"].append(semantic_path)
            paths_snow["dataset"].append("bdd100k")

        foggydriving_img_ids = [i_id.strip() for i_id in open(os.path.join(self.root, "ACG", "ACG_FoggyDriving_fog"))]
        for file_name in foggydriving_img_ids:
            file_path = os.path.join(self.root, "Foggy_Driving", file_name)
            if "test_extra" in file_name:
                semantic_name = file_name.replace("leftImg8bit/test_extra/", "gtCoarse/test_extra/").replace(
                    "_leftImg8bit.png", "_gtCoarse_labelTrainIds.png"
                )
            else:
                semantic_name = file_name.replace("leftImg8bit/test/", "gtFine/test/").replace(
                    "_leftImg8bit.png", "_gtFine_labelTrainIds.png"
                )
            semantic_path = os.path.join(self.root, "Foggy_Driving", semantic_name)
            paths_fog["image"].append(file_path)
            paths_fog["semantic"].append(semantic_path)
            paths_fog["dataset"].append("foggydriving")

        foggyzurich_img_ids = [i_id.strip() for i_id in open(os.path.join(self.root, "ACG", "ACG_FoggyZurich_fog"))]
        for file_name in foggyzurich_img_ids:
            file_path = os.path.join(self.root, "Foggy_Zurich", file_name)
            semantic_path = file_path.replace("/RGB/", "/gt_labelTrainIds/")
            paths_fog["image"].append(file_path)
            paths_fog["semantic"].append(semantic_path)
            paths_fog["dataset"].append("foggyzurich")

        custom_img_ids = [
            "train_000001.jpg",
            "train_000002.jpg",
            "train_000003.jpg",
        ]
        for file_name in custom_img_ids:
            file_path = os.path.join(self.root, "ACG", file_name)
            semantic_path = file_path.replace(".jpg", "_mask.png")
            paths_rain["image"].append(file_path)
            paths_rain["semantic"].append(semantic_path)
            paths_rain["dataset"].append("custom")

        self.len_fog = len(paths_fog["image"])
        self.len_night = len(paths_night["image"])
        self.len_rain = len(paths_rain["image"])
        self.len_snow = len(paths_snow["image"])

        for c in self.conditions:
            if c == "fog":
                self.paths["image"].extend(paths_fog["image"])
                self.paths["semantic"].extend(paths_fog["semantic"])
                self.paths["dataset"].extend(paths_fog["dataset"])
            elif c == "night":
                self.paths["image"].extend(paths_night["image"])
                self.paths["semantic"].extend(paths_night["semantic"])
                self.paths["dataset"].extend(paths_night["dataset"])
            elif c == "rain":
                self.paths["image"].extend(paths_rain["image"])
                self.paths["semantic"].extend(paths_rain["semantic"])
                self.paths["dataset"].extend(paths_rain["dataset"])
            elif c == "snow":
                self.paths["image"].extend(paths_snow["image"])
                self.paths["semantic"].extend(paths_snow["semantic"])
                self.paths["dataset"].extend(paths_snow["dataset"])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # load image
        # image = Image.open(self.paths[k][index]).convert("RGB")
        image = Image.open(self.paths["image"][index]).convert("RGB")

        # load semantic
        # semantic = Image.open(self.paths[k][index])
        semantic = Image.open(self.paths["semantic"][index])
        if self.paths["dataset"][index] == "wilddash":
            # print("Wilddash")
            filename = self.paths["image"][index].split("/")[-1]
            semantic = self.encode_semantic_map(semantic, filename.replace(".jpg", ".png"))

        if self.transforms is not None:
            image, semantic = self.transforms(image, semantic)

        semantic[semantic == 255] = 19

        return {"image_adv": image, "gt_adv": semantic}
        # return image, semantic

    def __len__(self) -> int:
        return len(self.paths["image"])

    def encode_semantic_map(self, semseg, filename):
        pan_format = np.array(semseg, dtype=np.uint32)
        pan = self.rgb2id(pan_format)
        semantic = np.zeros(pan.shape, dtype=np.uint8)
        for segm_info in self.img_to_segments[filename]:
            semantic[pan == segm_info["id"]] = segm_info["category_id"]
        semantic = self.id_to_train_id[semantic.astype(int)]
        return Image.fromarray(semantic.astype(np.uint8))

    @staticmethod
    def rgb2id(color):
        if color.dtype == np.uint8:
            color = color.astype(np.uint32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
