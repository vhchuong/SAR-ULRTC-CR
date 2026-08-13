# SAR-ULRTC-CR

Official PyTorch implementation of the paper:

> Chuong Hoang Vo and Chul Lee, “Unrolled Low-Rank Tensor Completion for SAR-Guided Cloud Removal in Multispectral Images,” *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, pp. 1–18, 2026, doi: [10.1109/JSTARS.2026.3723469](https://doi.org/10.1109/JSTARS.2026.3723469).

## Overview

SAR-ULRTC-CR addresses cloud removal in multispectral satellite imagery through an unrolled low-rank tensor-completion framework. The implementation provides dataset-specific network variants for SEN12MS-CR and SMILE-CR, together with preprocessing notebooks, SAR-to-optical translation resources, training and evaluation programs, and pretrained model parameters.

The released pipeline operates on cloudy multispectral observations, Sentinel-1 SAR data, and a three-channel translated image used as an auxiliary prior. The optimization procedure is unfolded into a fixed number of trainable stages; the default experiments use ten stages (`--N_iter 10`).

## Publication

- **Title:** [Unrolled Low-Rank Tensor Completion for SAR-Guided Cloud Removal in Multispectral Images](https://doi.org/10.1109/JSTARS.2026.3723469)
- **Authors:** Chuong Hoang Vo and Chul Lee
- **Journal:** IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
- **Year:** 2026
- **Pages:** 1–18
- **DOI:** [10.1109/JSTARS.2026.3723469](https://doi.org/10.1109/JSTARS.2026.3723469)

If this code or the associated models contribute to your research, please cite the paper using the entry in [Citation](#citation).

## Implementation

The network definitions are provided in [`model/main_net.py`](model/main_net.py). Two configurations are included:

| Dataset | Spectral bands | Model class | Training program | Evaluation program | Pretrained checkpoint |
| --- | ---: | --- | --- | --- | --- |
| SEN12MS-CR | 13 | `SAR_ULRTC_CR` | `model/train_SEN12MSCR.py` | `model/test_SEN12MSCR.py` | `ckpts_SEN12MSCR/epoch_200.pth` |
| SMILE-CR | 6 | `SAR_ULRTC_CR_SMILECR` | `model/train_SMILECR.py` | `model/test_SMILECR.py` | `ckpts_SMILECR/epoch_100.pth` |

### Repository organization

```text
.
├── ckpts_SEN12MSCR/          # SEN12MS-CR checkpoint at epoch 200
├── ckpts_SMILECR/            # SMILE-CR checkpoint at epoch 100
├── dataset/
│   ├── SEN12MSCR_dataset.ipynb
│   ├── SMILECR_dataset.ipynb
│   ├── dataLoader.py         # Original SEN12MS-CR data loaders
│   └── CycleGAN/             # Translation code, notebook, and model parameters
├── model/
│   ├── main_net.py           # Unrolled network definitions
│   ├── basicblock.py         # Convolutional and attention modules
│   ├── dataLoader.py         # SEN12MS-CR loaders used by the programs
│   ├── smilecr_dataset.py    # SMILE-CR NPY loaders
│   ├── train_SEN12MSCR.py
│   ├── test_SEN12MSCR.py
│   ├── train_SMILECR.py
│   ├── test_SMILECR.py
│   └── util/                 # Metrics, visualization, and data utilities
├── environment.yaml
└── LICENSE
```

## Requirements

The supplied Conda specification uses the following principal dependencies:

- Python 3.9
- PyTorch 2.2.2
- CUDA 11.8
- NumPy, SciPy, scikit-image, OpenCV, and Rasterio
- Matplotlib, tqdm, Visdom, TorchInfo, and THOP

Create the environment with:

```bash
conda env create -f environment.yaml
conda activate SAR-ULRTC-CR
```

The current implementation explicitly allocates CUDA tensors and calls `.cuda()`. An NVIDIA GPU with a compatible CUDA runtime is therefore required.

The training and evaluation commands should be executed from `model/`, because the programs use local imports and relative default paths:

```bash
cd model
```

## Data preparation

The raw datasets are not distributed in this repository. The following notebooks describe the preprocessing workflows:

- [`dataset/SEN12MSCR_dataset.ipynb`](dataset/SEN12MSCR_dataset.ipynb)
- [`dataset/SMILECR_dataset.ipynb`](dataset/SMILECR_dataset.ipynb)
- [`dataset/CycleGAN/CycleGAN.ipynb`](dataset/CycleGAN/CycleGAN.ipynb), used to generate the translated-image prior

Pretrained CycleGAN parameters are available in `dataset/CycleGAN/checkpoints/cyclegansen12/`.

### SEN12MS-CR

The SEN12MS-CR loaders require identical sample names across the cloudy input, cloud-free target, SAR, mask, and translated-image directories. The base NPY organization is:

```text
SEN12MS-CR_1/
├── SEN12MSCR_train_input_npy/
├── SEN12MSCR_train_gt_npy/
├── SEN12MSCR_train_sar_npy/
├── SEN12MSCR_train_mask_npy/
├── SEN12MSCR_test_input_npy/
├── SEN12MSCR_test_gt_npy/
├── SEN12MSCR_test_sar_npy/
├── SEN12MSCR_test_mask_npy/
├── SEN12MSCR_val_input_npy/
├── SEN12MSCR_val_gt_npy/
├── SEN12MSCR_val_sar_npy/
└── SEN12MSCR_val_mask_npy/
```

For `--data_mode full`, the training and test directory names end in `_npy_full` rather than `_npy`. The validation loader uses the unsuffixed directories shown above.

The two active SEN12MS-CR loaders use different translated-image formats:

- `SEN12MSCR_0`, used for training, reads PNG images from `SEN12MSCR_<split>_translated/` or `SEN12MSCR_<split>_translated_full/`.
- `SEN12MSCR_1`, used for evaluation, reads NPY arrays from `SEN12MSCR_<split>_translated_partial_npy/` or `SEN12MSCR_<split>_translated_full_npy/`.

### SMILE-CR

The SMILE-CR loaders expect the following split hierarchy:

```text
SMILE-CR-npy/
├── TrainData/TrainData/
├── ValData/ValData/
└── TestData/TestData/
```

Each final split directory contains:

```text
CloudLandsat_2020/       # Cloudy six-band NPY inputs
Landsat-8_2020/          # Cloud-free six-band NPY targets
MODIS_2020/              # Optional MODIS NPY data
Mask/                    # Cloud-mask NPY arrays
Sentinel-1_2020-De/      # Two-channel SAR NPY arrays
```

`SMILECRNumpyDataset`, used for training, reads translated PNG images from `Translated/`. `SMILECRNumpyDataset_1`, used for evaluation, reads translated NPY arrays from the split-specific `SMILECR_<split>_translated_full_npy/` directory.

## Training

### SEN12MS-CR

```bash
cd model
python train_SEN12MSCR.py \
  --train_path /path/to/SEN12MS-CR_1 \
  --save_path ../ckpts_SEN12MSCR \
  --data_mode full \
  --num_epoch 200 \
  --batch_size 1 \
  --N_iter 10 \
  --omega 0.5
```

When `--data_mode partial` is selected, the program randomly selects 5,000 training samples and writes their indices to `model/seed/train_5k_subset_indices.pth`. Create `model/seed/` before using this mode.

### SMILE-CR

```bash
cd model
python train_SMILECR.py \
  --train_path /path/to/SMILE-CR-npy \
  --save_path ../ckpts_SMILECR \
  --num_epoch 100 \
  --batch_size 1 \
  --N_iter 10 \
  --w 1.0
```

Both training programs accept `--resume /path/to/epoch_N.pth`, `--checkpoint_freq`, `--loss_freq`, and `--set_lr`. Each saved checkpoint contains the epoch number, model state, and optimizer state. The selected checkpoint directory also receives `learning_curve.png` and `training_log.txt`.

## Evaluation

### SEN12MS-CR

```bash
cd model
python test_SEN12MSCR.py \
  --data_path /path/to/SEN12MS-CR_1 \
  --ckpt_dir ../ckpts_SEN12MSCR \
  --epoch 200 \
  --N_iter 10 \
  --data_mode full \
  --gpu 0 \
  --fig_mode all
```

The SEN12MS-CR evaluation program processes the checkpoint specified by `--epoch`. Passing `--params` additionally reports the parameter count and estimated floating-point operations through THOP.

### SMILE-CR

```bash
cd model
python test_SMILECR.py \
  --data_path /path/to/SMILE-CR-npy \
  --ckpt_dir ../ckpts_SMILECR \
  --epoch 100 \
  --N_iter 10 \
  --gpu 0 \
  --fig_mode all \
  --test_mode single
```

For SMILE-CR, `--test_mode all --skip_epochs N` evaluates all checkpoints at or after epoch `N` and identifies the best checkpoints according to PSNR, SAM, and SSIM.

Set `--fig_mode none` to compute numerical results without saving visualizations. Otherwise, results are written to `model/results_SEN12MSCR/` or `model/results_SMILECR/`.

The evaluation programs report the following criteria:

- root mean squared error (RMSE)
- mean absolute error (MAE)
- peak signal-to-noise ratio (PSNR)
- spectral angle mapper (SAM), in degrees
- structural similarity index measure (SSIM)

## Reproducibility notes

- The pretrained checkpoints correspond to ten unrolled stages. Use `--N_iter 10` when evaluating the supplied parameters.
- The current implementation supports CUDA execution only.
- The model interfaces accept a direct SAR tensor. In the present network implementation, SAR guidance is incorporated through the translated-image prior; the direct SAR argument is retained by the interface but is not referenced by the unrolled blocks.
- Dataset files are not included and must be prepared using the supplied notebooks and the directory conventions above.

## Citation

```bibtex
@article{sar-ulrtc-cr,
  author  = {Vo, Chuong Hoang and Lee, Chul},
  title   = {Unrolled Low-Rank Tensor Completion for SAR-Guided Cloud Removal in Multispectral Images},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year    = {2026},
  pages   = {1--18},
  doi     = {10.1109/JSTARS.2026.3723469}
}
```

## License

This software is distributed under the [MIT License](LICENSE).
