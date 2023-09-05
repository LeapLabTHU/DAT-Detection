# Vision Transformer with Deformable Attention

This repository contains the code of object detection and instance segmentation for the paper Vision Transformer with Deformable Attention \[[arXiv](https://arxiv.org/abs/2201.00520)\], and DAT++: Spatially Dynamic Vision Transformerwith Deformable Attention (extended version)\[[OneDrive](https://1drv.ms/b/s!ApI0vb6wPqmtgrl6Pqn0wybDrpaxvg?e=4yVs7Z)]. 

This code is based on [mmdetection](https://github.com/open-mmlab/mmdetection) and [Swin Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection). To get started, you can follow the instructions in [Swin Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/README.md).

Other links:

- [Classification](https://github.com/LeapLabTHU/DAT)
- [Segmentation](https://github.com/LeapLabTHU/DAT-Segmentation)

## Dependencies

In addition to the dependencies of the [classification](https://github.com/LeapLabTHU/DAT) codebase, the following packages are required:

- mmcv-full == 1.4.0
- mmdetection == 2.26.0

## Evaluating Pretrained Models

### RetinaNet

| Backbone | Schedule  | bbox mAP | mask mAP | config | pretrained weights |
| :---: | :---: | :---: | :---: | :---: | :---: |
| DAT-T++ | 1x | 46.8 | - | [config](configs/dat/rtn_tiny_1x_4n_dp00_lr2.py) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroLKGEtPqZe5vVKgA?e=l61dNf) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/8e49b993adf242829ee1/) |
| DAT-T++ | 3x | 49.2 | - | [config](configs/dat/rtn_tiny_3x_4n_dp02_lr2.py) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroDwb1rL_Fb3ZFBWg?e=SB0b4R) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/ff4e48c036cb4d9badca/) |
| DAT-S++ | 1x | 48.3 | - | [config](configs/dat/rtn_small_1x_4n_dp01_lr2.py) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroKRvxYPXTxsaaLUg?e=9Ty1zb) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/c1585651b3ec41fa9eb5/) |
| DAT-S++ | 3x | 50.2 | - | [config](configs/dat/rtn_small_3x_4n_dp05_lr2.py) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroOnfDaAaodgfU9og?e=DuLtLB) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/3687b77098a647adb54f/) |

### Mask R-CNN

| Backbone | Schedule | bbox mAP | mask mAP | config | pretrained weights |
| :---: | :---: | :---: | :---: | :---: | :---: |
| DAT-T++ | 1x | 48.7 | 43.7 | [config](configs/dat/mrcn_tiny_1x_4n_dp00_lr4.py) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroIh3FpJJ1JezkAdw?e=ripLe0) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/2b61106279ae488bbb45/) |
| DAT-T++ | 3x | 50.5 | 45.1 | [config](configs/dat/mrcn_tiny_3x_4n_dp03_lr4.py) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroMWHVicveA4DjkFQ?e=b76PGk) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/a7db396a372449779347/) |
| DAT-S++ | 1x | 49.8 | 44.5 | [config](configs/dat/mrcn_small_1x_4n_dp01_lr4.py) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroFCpEZ1rCavI2xFg?e=O9w2ff) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/8f3653c027974136a74a/) |
| DAT-S++ | 3x | 51.2 | 45.7 | [config](configs/dat/mrcn_small_3x_4n_dp05_lr4.py) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroH1ZALwoxhcO37MQ?e=2aZeBj) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/3bde5a98a0ca46d8b2ee/) |

### Cascade Mask R-CNN

| Backbone | Schedule | bbox mAP | mask mAP | config | pretrained weights |
| :---: | :---: | :---: | :---: | :---: | :---: |
| DAT-T++ | 1x | 52.2 | 45.0 | [config](configs/dat/cmrcn_tiny_1x_4n_dp00_lr4.py) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroGo2oubInWIwrnWg?e=o9aJlK) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/35be35faa96044ee8561/) |
| DAT-T++ | 3x | 53.0 | 46.0 | [config](configs/dat/cmrcn_tiny_3x_4n_dp01_lr4.py) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroEOC8rgHFWAnO2yw?e=2BLK8g) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/1aa96891dda24451bbf8/) |
| DAT-S++ | 3x | 54.2 | 46.9 | [config](configs/dat/cmrcn_small_3x_4n_dp05_lr4.py) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroJ0--lDvJw90u2_g?e=293ING) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/17d84d4f5ae2457285af/) |
| DAT-B++ | 3x | 54.5 | 47.0 | [config](configs/dat/cmrcn_base_3x_8n_dp08_lr4.py) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroNdQXStfHGPbzkpw?e=fWJT5O) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/9ea06105dd394b8d97ac/) |


To evaluate a pretrained checkpoint, please download the pretrain weights to your local machine and run the mmdetection test scripts as follows:

```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
bash tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

**Please notice: Before training or evaluation, please set the `data_root` variable in `configs/_base_/datasets/coco_detection.py` (RetinaNet) and `configs/_base_/datasets/coco_instance.py` (Mask R-CNN & Cascade Mask R-CNN) to the path where MS-COCO data stores.**

Since evaluating models needs no pretrain weights, you can set the `pretrained = None` in `<CONFIG_FILE>`.

## Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE>

# multi-gpu training
bash tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> 
```

**Please notice: Make sure the `pretrained` variable in `<CONFIG_FILE>` is correctly set to the path of pretrained DAT model.**

In our experiments, we typically use 4 nodes of NVIDIA A100 GPU (40GB) to train the models, so the learning rates are scaled to 4 times of the default values for each detector.

## Acknowledgements

This code is developed on the top of [Swin Transformer](https://github.com/microsoft/Swin-Transformer), we thank to their efficient and neat codebase. The computational resources supporting this work are provided by [Hangzhou
High-Flyer AI Fundamental Research Co.,Ltd](https://www.high-flyer.cn/).

## Citation

If you find our work is useful in your research, please consider citing:

```
@InProceedings{Xia_2022_CVPR,
    author    = {Xia, Zhuofan and Pan, Xuran and Song, Shiji and Li, Li Erran and Huang, Gao},
    title     = {Vision Transformer With Deformable Attention},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {4794-4803}
}
```

## Contact

If you have any questions or concerns, please send email to [xzf23@mails.tsinghua.edu.cn](mailto:xzf23@mails.tsinghua.edu.cn).


