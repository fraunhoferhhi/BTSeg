from pathlib import Path

import pytorch_lightning as pl
import torch
from numpy import load
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

from src import utils
from src.data.data_transforms import (  # RandomScale,
    Compose,
    MutualRandomCrop,
    Normalize,
    RandomHorizontalFlip,
    Resize,
)
from src.utils.cityscape_utils import id2label

log = utils.get_pylogger(__name__)

# ACDC classes (like Cityscapes train ids)
# 'unlabeled'             19
# 'road'                   0
# 'sidewalk'               1
# 'building'               2
# 'wall'                   3
# 'fence'                  4
# 'pole'                   5
# 'traffic light'          6
# 'traffic sign'           7
# 'vegetation'             8
# 'terrain'                9
# 'sky'                   10
# 'person'                11
# 'rider'                 12
# 'car'                   13
# 'truck'                 14
# 'bus'                   15
# 'train'                 16
# 'motorcycle'            17
# 'bicycle'               18


class ACDCDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        mode: str,
        weather_condition: str,
        transform: Compose,
        single_mode: bool = False,
        use_warped_target: bool = False,
        use_crop_warped_target: bool = None,
    ) -> None:
        """Creates a PyTorch Dataset for the ACDC dataset.

        Args:
            data_dir (Path): Data root path
            mode (str): Defines the dataset split to load. ["train", "val", "pred"]
            weather_condition (str): Defines the weather condition to load. [ "fog", "nigth", "rain", "snow", "clear", "all"]
            transform (Compose): Transformations to apply to the dataset.
            single_mode (bool): Defines if only the selected weather condition should be loaded (NOT combining clear + selcected condition). Default: False
            use_warped_target (bool): Defines if the warped target should be used. Default: False
        Raises:
            RuntimeError: For unknown mode options.
            RuntimeError: If the the data_dir can't be find.
        """
        super().__init__()

        self.data_dir = data_dir
        self.mode = mode
        self.weather_condition = weather_condition
        self.single_mode = single_mode
        self.use_warped_target = use_warped_target
        self.use_crop_warped_target = (
            use_crop_warped_target if use_crop_warped_target is not None else use_warped_target
        )

        if self.use_warped_target:
            assert self.mode == "train", "Warped target can only be used in training mode!"
        if self.use_crop_warped_target:
            assert use_warped_target, "Crop warped target can only be used if warped target is used!"
        if self.use_warped_target and not self.use_crop_warped_target:
            log.warning("Warped target is used but not cropped!")

        # fmt: off
        # exclude samples from the training set
        # superset of images smaller than 768x768 and images with 0 valid pixels/ patches according to 30_ACDC_statistics_uncertainty_patches.ipynb
        self.ids_samples_to_exclude = {"rain": [34, 41, 48, 48, 52, 55, 70, 101, 124, 164, 168, 169, 186, 190, 190, 191, 194, 194, 207, 308, 308, 316, 321, 322, 330, 349, 350, 350, 351, 351, 352, 353, 354, 375, 377, 380, 380, 381, 383, 384, 387, 388, 388, 389, 390, 391, 394],
                                       "night": [0, 1, 10, 22, 33, 36, 36, 54, 56, 56, 57, 70, 72, 84, 89, 94, 126, 143, 145, 149, 159, 160, 180, 184, 189, 190, 190, 205, 206, 228, 249, 251, 252, 252, 261, 261, 300, 314, 319, 322, 330, 332, 336, 343, 352, 356, 360, 362, 365, 366, 373, 374, 378, 379, 381, 383, 384, 388, 395, 396, 399],
                                       "fog": [0, 1, 2, 5, 6, 7, 8, 16, 24, 43, 44, 48, 49, 51, 52, 53, 54, 66, 78, 79, 80, 81, 82, 83, 84, 85, 86, 88, 89, 90, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 114, 115, 116, 117, 124, 125, 128, 129, 130, 132, 133, 141, 142, 143, 144, 147, 160, 176, 178, 181, 182, 188, 189, 190, 191, 192, 193, 231, 232, 233, 234, 236, 237, 244, 246, 247, 248, 255, 260, 263, 264, 265, 266, 267, 268, 269, 272, 299, 300, 304, 305, 306, 307, 308, 310, 311, 312, 313, 313, 315, 316, 317, 318, 319, 335, 337, 339, 340, 341, 342, 343, 344, 379, 384, 386, 392],
                                       "snow": [0, 1, 2, 4, 17, 18, 19, 20, 46, 47, 48, 48, 101, 104, 104, 105, 105, 106, 107, 107, 109, 110, 111, 112, 113, 115, 120, 171, 179, 184, 185, 196, 199, 200, 207, 210, 216, 217, 221, 226, 227, 228, 231, 233, 248, 251, 253, 254, 255, 256, 257, 258, 259, 260, 261, 265, 271, 273, 279, 285, 290, 291, 292, 293, 295, 307, 308, 309, 311, 312, 313, 314, 315, 316, 342, 343, 344, 344, 345, 345, 346, 346, 347, 348, 355, 357, 358, 358, 359, 360, 372, 373, 376, 384, 384, 387]}
        # fmt: on

        if transform is None:
            self.transform = Compose([])
        else:
            self.transform = transform

        if self.mode not in ["train", "val", "test", "pred"]:
            raise RuntimeError(
                "Unknown option " + self.mode + " as mode variable. Valid options: 'train', 'val', 'test' and 'pred'"
            )

        if self.mode == "pred":  # prediction uses the test set
            self.mode = "test"

        weather_conditions = ["fog", "night", "rain", "snow"]

        if self.weather_condition not in weather_conditions + ["clear", "all"]:
            raise RuntimeError(
                "Unknown option "
                + self.weather_condition
                + " as weather condition variable. Valid options: 'fog', 'night', 'rain', 'snow', 'clear' and 'all'"
            )

        if not self.data_dir.is_dir():
            raise RuntimeError(str(self.data_dir) + " is not a directory!")

        self.weather_condition_query = (
            weather_conditions if self.weather_condition in ["clear", "all"] else [self.weather_condition]
        )

        self.ReadSampleFiles()
        self.ReadGTFiles()

        # self.SanityCheck()

        log.info(f"Found {len(self)} samples in {self.data_dir}.")

    def ReadSampleFiles(self):
        """Read the adverse weather images.

        Depends on the mode parameter.
        """

        file_name_pattern_ref = "_ref_anon.png"

        if self.use_warped_target:
            file_name_pattern = "_rgb_anon_warped.png"
            self.warp_conf_files = []
            self.intrinsic_rect_files = []
        else:
            file_name_pattern = "_rgb_anon.png"

        self.rgb_files = []
        self.rgb_files_ref = []

        for weather_condition in self.weather_condition_query:
            rgb_files = sorted(
                list(
                    self.data_dir.glob("rgb_anon/" + weather_condition + "/" + self.mode + "/**/*" + file_name_pattern)
                ),
                key=lambda i: i.stem[:21],
            )

            rgb_files_ref = sorted(
                list(
                    self.data_dir.glob(
                        "rgb_anon/" + weather_condition + "/" + self.mode + "_ref" + "/**/*" + file_name_pattern_ref
                    )
                ),
                key=lambda i: i.stem[:21],
            )

            if self.use_warped_target:
                warp_conf_files = sorted(
                    list(
                        self.data_dir.glob(
                            "rgb_anon/" + weather_condition + "/" + self.mode + "/**/*_rgb_anon_uncertainty.npy"
                        )
                    ),
                    key=lambda i: i.stem[:21],
                )

                intrinsic_rect_files = sorted(
                    list(
                        self.data_dir.glob(
                            "rgb_anon/" + weather_condition + "/" + self.mode + "/**/*_intrinsic_rect.npy"
                        )
                    ),
                    key=lambda i: i.stem[:21],
                )

                if self.mode not in ["test", "pred", "val"]:
                    rgb_files = self.exlude_samples(rgb_files, self.ids_samples_to_exclude[weather_condition])
                    rgb_files_ref = self.exlude_samples(rgb_files_ref, self.ids_samples_to_exclude[weather_condition])
                    warp_conf_files = self.exlude_samples(
                        warp_conf_files, self.ids_samples_to_exclude[weather_condition]
                    )
                    intrinsic_rect_files = self.exlude_samples(
                        intrinsic_rect_files, self.ids_samples_to_exclude[weather_condition]
                    )

                self.warp_conf_files += warp_conf_files
                self.intrinsic_rect_files += intrinsic_rect_files

            self.rgb_files += rgb_files
            self.rgb_files_ref += rgb_files_ref

    def ReadGTFiles(self):
        """Reads the groundtruth segmentation."""

        self.gt_files = []
        self.gt_files_ref = []

        for weather_condition in self.weather_condition_query:
            if self.mode not in ["test", "pred"]:
                gt_files = sorted(
                    list(self.data_dir.glob("gt/" + weather_condition + "/" + self.mode + "/**/*labelTrainIds.png")),
                    key=lambda i: i.stem[:21],
                )

            gt_files_ref = sorted(
                list(
                    self.data_dir.glob("gt/" + weather_condition + "/" + self.mode + "_ref" + "/**/*labelTrainIds.png")
                ),
                key=lambda i: i.stem[:21],
            )

            if self.mode not in ["test", "pred"]:
                if self.mode not in ["test", "pred", "val"] and self.use_warped_target:
                    gt_files = self.exlude_samples(gt_files, self.ids_samples_to_exclude[weather_condition])
                self.gt_files += gt_files

            if self.mode not in ["test", "pred", "val"] and self.use_warped_target:
                gt_files_ref = self.exlude_samples(gt_files_ref, self.ids_samples_to_exclude[weather_condition])
            self.gt_files_ref += gt_files_ref

    @staticmethod
    def exlude_samples(list_samples, list_ids_to_filter):
        list_samples = [f for i, f in enumerate(list_samples) if i not in list_ids_to_filter]

        return list_samples

    def SanityCheck(self):
        """Simple sanity checks whether the numbers are matching."""
        if self.mode in ["test", "pred"]:
            assert len(self.gt_files) == 0, f"GT files: {len(self.gt_files)}"
            assert (
                len(self.rgb_files) == len(self.rgb_files_ref) == len(self.gt_files_ref)
            ), f"RGB files: {len(self.rgb_files)} RGB ref files: {len(self.rgb_files_ref)} GT ref files: {len(self.gt_files_ref)}"
        else:
            assert (
                len(self.gt_files) == len(self.rgb_files) == len(self.gt_files_ref) == len(self.rgb_files_ref)
            ), f"GT files: {len(self.gt_files)} RGB files: {len(self.rgb_files)} RGB ref files: {len(self.rgb_files_ref)} GT ref files: {len(self.gt_files_ref)}"

            if self.use_warped_target:
                assert len(self.rgb_files) == len(
                    self.warp_conf_files
                ), f"RGB files: {len(self.rgb_files)} Warp conf files: {len(self.warp_conf_files)}"

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        path_rgb_file = self.rgb_files[index]
        path_rgb_ref_file = self.rgb_files_ref[index]
        path_gt_ref_file = self.gt_files_ref[index]
        assert path_rgb_file.stem[:21] == path_rgb_ref_file.stem[:21] == path_gt_ref_file.stem[:21]

        if self.use_warped_target:
            path_conf_file = self.warp_conf_files[index]
            assert path_rgb_file.stem[:21] == path_conf_file.stem[:21]

        rgb_image = read_image(str(path_rgb_file)).type(torch.float) / 255.0
        rgb_ref_image = read_image(str(path_rgb_ref_file)).type(torch.float) / 255.0
        gt_ref = read_image(str(path_gt_ref_file)).squeeze(0).type(torch.long)

        if self.use_warped_target:
            conf = torch.from_numpy(load(path_conf_file))
            rect = load(self.intrinsic_rect_files[index])

        if self.mode in ["test", "pred"]:
            gt = torch.ones((rgb_image.size()[1], rgb_image.size()[2]), dtype=torch.long) * 255
        else:
            path_gt_file = self.gt_files[index]

            assert path_gt_file.stem[:21] == path_rgb_file.stem[:21], "File mismatch!"

            gt = read_image(str(path_gt_file)).squeeze(0).type(torch.long)

        if self.use_warped_target and self.use_crop_warped_target:
            rgb_image = self.crop_intrinsic_rect_from_tensor(rgb_image, rect)
            rgb_ref_image = self.crop_intrinsic_rect_from_tensor(rgb_ref_image, rect)
            gt = self.crop_intrinsic_rect_from_tensor(gt, rect)
            gt_ref = self.crop_intrinsic_rect_from_tensor(gt_ref, rect)
            conf = self.crop_intrinsic_rect_from_tensor(conf, rect)

        if self.use_warped_target:
            gt = gt.type(torch.float)
            gt = torch.cat(
                [gt.unsqueeze_(0), conf.unsqueeze_(0)], dim=0
            )  # hacky way to transform the confidence map along with the gt and image

        images, gts = self.transform([rgb_ref_image, rgb_image], [gt_ref, gt])

        if self.use_warped_target:
            gt, conf = torch.tensor_split(gts[1], 2, dim=0)
            conf = conf.squeeze(0)
            gts[1] = gt.type(torch.long).squeeze(0)

        # self.replaceIdwithTrainId(gts[0])
        gts[0][gts[0] == 255] = 19

        # self.replaceIdwithTrainId(gts[1])
        gts[1][gts[1] == 255] = 19

        if self.single_mode:
            if self.weather_condition == "clear":
                sample = {"image": images[0], "gt": gts[0]}
            else:
                sample = {"image": images[1], "gt": gts[1]}
        else:
            if self.use_warped_target:
                sample = {
                    "image": images[0],
                    "gt": gts[0],
                    "image_adv": images[1],
                    "gt_adv": gts[1],
                    "conf": conf,
                    "rect": rect,
                }
            else:
                sample = {"image": images[0], "gt": gts[0], "image_adv": images[1], "gt_adv": gts[1]}

        return sample

    @staticmethod
    def crop_intrinsic_rect_from_tensor(image, rect):
        if image.dim() == 3:
            return image[:, rect[0][1] : rect[1][1], rect[0][0] : rect[1][0]]
        elif image.dim() == 2:
            return image[rect[0][1] : rect[1][1], rect[0][0] : rect[1][0]]
        else:
            raise RuntimeError("Invalid image dimensions!")

    def get_paths_sample(self, index):
        """Returns the paths of the sample."""

        if torch.is_tensor(index):
            index = index.tolist()

        path_rgb_file = self.rgb_files[index]
        path_rgb_ref_file = self.rgb_files_ref[index]
        path_gt_ref_file = self.gt_files_ref[index]

        assert path_rgb_file.stem[:21] == path_rgb_ref_file.stem[:21] == path_gt_ref_file.stem[:21], "File mismatch!"
        if self.use_warped_target:
            path_conf_file = self.warp_conf_files[index]
            assert path_rgb_file.stem[:21] == path_conf_file.stem[:21]

        if self.mode not in ["test", "pred", "train"]:
            path_gt_file = self.gt_files[index]

            assert path_gt_file.stem[:21] == path_rgb_file.stem[:21], "File mismatch!"
        else:
            path_gt_file = None

        if self.use_warped_target:
            return {
                "image": path_rgb_ref_file,
                "image_adv": path_rgb_file,
                "gt": path_gt_ref_file,
                "gt_adv": path_gt_file,
                "conf": path_conf_file,
            }
        else:
            return {
                "image": path_rgb_ref_file,
                "image_adv": path_rgb_file,
                "gt": path_gt_ref_file,
                "gt_adv": path_gt_file,
            }

    @staticmethod
    def replaceIdwithTrainId(gt):
        """Replaces the ids with the train ids."""
        for id, label in id2label.items():
            gt[gt == id] = label.trainId


class ACDCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        weather_condition: str,
        batch_size: int,
        num_workers: int,
        use_warped_target: bool = False,
        use_crop_warped_target: bool = None,
        val_batch_size: int = None,
        test_batch_size: int = None,
        shuffle_train_data: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last: bool = True,
        image_height: int = 512,
        image_width: int = 512,
        single_mode: bool = False,
        test_pred_on_val: bool = False,
        size_smaller_edge_val_test: int = 1080,
    ):
        """Creates a PyTorch Lightning Data Module for the ACDC dataset.

        Paper: https://arxiv.org/abs/2104.13395
        Website: https://acdc.vision.ee.ethz.ch/

        Args:
            data_dir (Path): Path to the root directory containing the data.
            weather_condition (str): Defines the weather condition to load. ["fog", "nigth", "rain", "snow", "clear", "all"]
            batch_size (int): batch size used
            num_workers (int): number of worker threads used for data loading
            use_warped_target (bool): whether to use the warped target. Default: False
            use_crop_warped_target (bool): whether to crop the warped target. Default: None, which uses the same value as use_warped_target.
            val_batch_size (int): batch size used for validation. Defaults to None, which uses the same batch size as training.
            test_batch_size (int): batch size used for testing. Defaults to None, which uses the same batch size as training.
            shuffle_train_data (bool): shuffle the training data. Default: True
            pin_memory (bool): pin memory for faster data transfer to GPU. Default: True
            persistent_workers (bool): keep the data loader workers alive. Default: True
            drop_last (bool): drop the last batch if it is smaller than the batch size. Default: True
            image_height (int): height images dataset after resize. Default: 512
            image_width (int): width images dataset after resize. Default: 512
            single_mode (bool): outputs only samples in the selected weather condition (NOT combining clear + selcected condition). Default: False
            test_pred_on_val (bool): whether to test/predict on the validation set for debugging. Default: False
            size_smaller_edge_val_test (int): size of the smaller edge of the image after resize for validation and test. Default: 1080 (original size)
        """

        super().__init__()
        self.save_hyperparameters(logger=True)

        self.hparams.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.hparams.test_batch_size = batch_size if test_batch_size is None else test_batch_size

        if isinstance(self.hparams.data_dir, str):
            self.hparams.data_dir = Path(self.hparams.data_dir)

        self.hparams.data_dir = self.hparams.data_dir / "ACDC"

        if image_height <= 0 or image_width <= 0:
            raise AttributeError("Invalid value for image_height or image_width. Both should be greater than zero.")

        self.transform_train = Compose(
            [
                MutualRandomCrop((768, 768), cat_max_ratio=1.0),
                Resize((self.hparams.image_height, self.hparams.image_width)),
                RandomHorizontalFlip(0.5),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.transform_val_test = Compose(
            [
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        if self.hparams.size_smaller_edge_val_test != 1080:
            log.warning(f"Resizing validation and test images to {self.hparams.size_smaller_edge_val_test} !")
            self.transform_val_test.transforms.insert(
                0, Resize((self.hparams.size_smaller_edge_val_test, self.hparams.size_smaller_edge_val_test))
            )

        if self.hparams.test_pred_on_val:
            log.warning("Testing/Prediction on validation set!")

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

        if stage == "fit" or stage is None:
            self.acdc_train = ACDCDataset(
                self.hparams.data_dir,
                "train",
                self.hparams.weather_condition,
                self.transform_train,
                single_mode=self.hparams.single_mode,
                use_warped_target=self.hparams.use_warped_target,
                use_crop_warped_target=self.hparams.use_crop_warped_target,
            )
            self.acdc_val = ACDCDataset(
                self.hparams.data_dir,
                "val",
                self.hparams.weather_condition,
                self.transform_val_test,
                single_mode=self.hparams.single_mode,
            )
        elif stage == "test":
            self.acdc_test = ACDCDataset(
                self.hparams.data_dir,
                "test" if not self.hparams.test_pred_on_val else "val",
                self.hparams.weather_condition,
                self.transform_val_test,
                single_mode=self.hparams.single_mode,
            )
        elif stage == "predict":
            self.acdc_predict = ACDCDataset(
                self.hparams.data_dir,
                "pred" if not self.hparams.test_pred_on_val else "val",
                self.hparams.weather_condition,
                self.transform_val_test,
                single_mode=self.hparams.single_mode,
            )
        else:
            raise RuntimeError(
                "Unknown option " + stage + " as stage variable. Valid options: 'fit', 'test', or 'predict'"
            )

    def train_dataloader(self):
        return DataLoader(
            self.acdc_train,
            num_workers=self.hparams.num_workers,
            shuffle=self.hparams.shuffle_train_data,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=False if not self.hparams.num_workers > 1 else self.hparams.persistent_workers,
            batch_size=self.hparams.batch_size,
            drop_last=self.hparams.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.acdc_val,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            drop_last=self.hparams.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            self.acdc_test,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.acdc_predict,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )
