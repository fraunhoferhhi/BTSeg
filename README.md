<div align="center">

## BTSeg: Barlow Twins Regularization for Domain Adaptation in Semantic Segmentation

## [Paper](https://arxiv.org/abs/2308.16819) | [Result ACDC](https://acdc.vision.ee.ethz.ch/submissions/65576fc7487cf3198c218e01)

<img src="./docs/Teaser_Image.png" width="500"/>

BTSeg employs a novel application of the Barlow Twins loss, a concept borrowed from unsupervised learning.
The original Barlow Twins approach uses stochastic augmentations in order to learn useful representations from unlabeled data without the need for external labels.
In our approach, we regard images captured at identical locations but under varying adverse conditions as manifold representation of the same scene (which could be interpreted as ”natural augmentations”), thereby enabling the model to conceptualize its understanding of the environment.

</div>

______________________________________________________________________

Official PyTorch implementation of the paper [BTSeg: Barlow Twins Regularization for Domain Adaptation in Semantic Segmentation](https://arxiv.org/abs/2308.16819).

The code is organized using [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

# Setup:

## 1. Create environment using conda:

```bash
 conda env create --file conda_env.yml
 conda activate btseg-env
```

## 2. Create an environment file (.env) and specify the location of the datasets and where to save the log files.

```bash
 DATA_DIR="/path/to/root/directory/data"
 OUTPUT_DIR="/path/to/output/dir"
```

## 3. Download the required pretrained weights.

- [SegFormer](https://github.com/NVlabs/SegFormer) weights pretrained on Cityscapes (segformer.b5.1024x1024.city.160k.pth)
- [UAWarpC](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/626140/uawarpc_megadepth.ckpt) pretrained on MegaDepth
- save them in ./pretrained_weights/

## 4. Download checkpoint

- download the [checkpoint](https://cvg.hhi.fraunhofer.de/BTSeg/checkpoints/BTSeg_ACDC_best.ckpt) of our model used for comparison with the state of the art
- store it in ./checkpoints/

## 4. Prepare the ACDC data

- Register and download the data from https://acdc.vision.ee.ethz.ch/

- The scripts assume that the folder is named "ACDC" and is located under the path specified in DATA_DIR

- Download the [segmentation maps for the reference images](https://cvg.hhi.fraunhofer.de/BTSeg/label_ref/ACDC_label_ref.zip) and merge them into the ACDC root folder

  - generated using [Hierarchical Multi-Scale Attention for Semantic Segmentation](https://github.com/segcv/hierarchical-multi-scale-attention)

- Final structure:

  - /path/to/ACDC
    - gt
      - fog
        - *test_ref*
        - train
        - *train_ref*
        - val
        - *val_ref*
      - night
      - ...
    - rgb_anon
      - fog
      - night
      - ...

- Run warp_target_images.py to align the target (adverse weather conditions) to the source (reference weather conditions) images

  ```bash
  python warp_target_images.py
  ```

- Generates a \*\_rgb_anon_warped.png for every \*\_rgb_anan.png image

## 5. (Optional) Create the ACG benchmark dataset.

- Download the list of filenames and follow the instructions from [here](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/626144/ACG.zip)
- name the folder "ACG_benchmark"

# Test:

On the ACDC validation subset (should result in a mean IoU for the adverse images of 0.69925):

```bash
python src/run.py test -c ./configs/Test_BT-Seg_ACDC.yaml --ckpt_path ./checkpoints/BTSeg_ACDC_best.ckpt --data.test_pred_on_val True
```

Evaluation on rainy images only:

```bash
python src/run.py test -c ./configs/Test_BT-Seg_ACDC.yaml --ckpt_path ./checkpoints/BTSeg_ACDC_best.ckpt --data.test_pred_on_val True --data.weather_condition rain
```

On the ACG benchmark dataset:

```bash
python src/run.py test -c ./configs/Test_BT-Seg_ACG.yaml --ckpt_path ./checkpoints/BTSeg_ACDC_best.ckpt
```

## Do prediction on test data:

```bash
python src/run.py predict -c ./configs/Test_BT-Seg_ACDC.yaml --ckpt_path ./checkpoints/BTSeg_ACDC_best.ckpt --model.predict_mode png
# model.predict_mode matplotlib is also possible, to generate easy readable segmentation maps + images
python src/run.py predict -c ./configs/Test_BT-Seg_ACDC.yaml --ckpt_path ./checkpoints/BTSeg_ACDC_best.ckpt --data.test_pred_on_val True --model.predict_mode matplotlib
```

# Training:

```bash
python src/run.py fit -c ./configs/Train_BT-Seg_ACDC.yaml
# with smaller input images
python src/run.py fit -c ./configs/Train_BT-Seg_ACDC.yaml --data.image_height 512 --data.image_width 512
```

# Citation

If you find this code useful in your research, please consider citing the paper:

```bibtex
@misc{btseg,
      title={BTSeg: Barlow Twins Regularization for Domain Adaptation in Semantic Segmentation},
      author={Johannes K{\"u}nzel and Anna Hilsmann and Peter Eisert},
      year={2023},
      eprint={2308.16819},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# License

This repository is released under the MIT license. However, care should be taken to adopt the appropriate license for third-party code in this repository. Third-party code is marked as such.

