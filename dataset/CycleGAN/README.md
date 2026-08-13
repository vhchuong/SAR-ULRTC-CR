# CycleGAN Translation for SAR-ULRTC-CR

This directory contains the CycleGAN inference and postprocessing pipeline used to produce the three-channel translated-image prior for SAR-ULRTC-CR. The workflow follows [`CycleGAN.ipynb`](CycleGAN.ipynb) and supports the SEN12MS-CR and SMILE-CR datasets.

The code is adapted from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). The project-specific entrypoint is [`test_cloud_SEN12.py`](test_cloud_SEN12.py), and pretrained parameters are provided in `checkpoints/cyclegansen12/`.

## Workflow

The translation workflow consists of two stages:

1. Run CycleGAN inference for the required dataset split and data mode.
2. Rename the generated `*_fake_A.npy` files, collect them in a `*_translated_*_npy` directory, and move that directory into the corresponding dataset tree.

Run all commands from this directory:

```bash
cd dataset/CycleGAN
```

## Requirements

Install the main project environment and the CycleGAN-specific dependencies:

```bash
conda env create -f ../../environment.yaml
conda activate SAR-ULRTC-CR
pip install -r requirements.txt
```

The supplied experiment is named `cyclegansen12`. Its checkpoint directory contains:

```text
checkpoints/cyclegansen12/
├── latest_net_G_A.pth
├── latest_net_G_B.pth
├── latest_net_D_A.pth
└── latest_net_D_B.pth
```

Inference loads the two `latest_net_G_*.pth` generator files. The discriminator files are retained with the complete trained experiment.

## Inference

Set `--dataroot` to the dataset root and keep `--name` consistent with the checkpoint directory. The notebook uses the `cyclegansen12` experiment, the `cycle_gan` model, and the `BtoA` direction.

### SEN12MS-CR

```bash
python test_cloud_SEN12.py \
  --dataroot /path/to/SEN12MS-CR_1 \
  --name cyclegansen12 \
  --model cycle_gan \
  --direction BtoA \
  --gpu_ids 1 \
  --data_mode full \
  --dataset SEN12MSCR \
  --mode test
```

For partial data, replace `--data_mode full` with `--data_mode partial`. The active loader expects matching NPY filenames in the following directories:

| Split | Full mode | Partial mode |
| --- | --- | --- |
| Train | `SEN12MSCR_train_input_npy_full`, `SEN12MSCR_train_gt_npy_full`, `SEN12MSCR_train_sar_npy_full` | `SEN12MSCR_train_input_npy`, `SEN12MSCR_train_gt_npy`, `SEN12MSCR_train_sar_npy` |
| Test | `SEN12MSCR_test_input_npy_full`, `SEN12MSCR_test_gt_npy_full`, `SEN12MSCR_test_sar_npy_full` | `SEN12MSCR_test_input_npy`, `SEN12MSCR_test_gt_npy`, `SEN12MSCR_test_sar_npy` |
| Validation | `SEN12MSCR_val_input_npy`, `SEN12MSCR_val_gt_npy`, `SEN12MSCR_val_sar_npy` | Same as full mode |

Select the split with `--mode train`, `--mode test`, or `--mode val`.

### SMILE-CR

```bash
python test_cloud_SEN12.py \
  --dataroot /path/to/SMILE-CR-npy \
  --name cyclegansen12 \
  --model cycle_gan \
  --direction BtoA \
  --gpu_ids 1 \
  --data_mode full \
  --dataset SMILECR \
  --mode test
```

The SMILE-CR loader selects the split through `--mode` and expects this hierarchy:

```text
SMILE-CR-npy/
├── TrainData/TrainData/
├── ValData/ValData/
└── TestData/TestData/
```

Each final split directory contains matching NPY files in:

```text
CloudLandsat_2020/
Landsat-8_2020/
MODIS_2020/
Mask/
Sentinel-1_2020-De/
```

For SMILE-CR, `--data_mode` is retained in the output-directory name; split selection is controlled by `--mode`.

### Principal arguments

| Argument | Description |
| --- | --- |
| `--dataroot` | Root of the prepared SEN12MS-CR or SMILE-CR dataset |
| `--name` | Experiment/checkpoint directory name under `checkpoints/` |
| `--dataset` | `SEN12MSCR` or `SMILECR` |
| `--mode` | Dataset split: `train`, `val`, or `test` |
| `--data_mode` | `full` or `partial`; also forms part of the output name |
| `--direction` | Translation direction used by the checkpoint; the notebook uses `BtoA` |
| `--gpu_ids` | CUDA device ID; the notebook uses `1`. Select an available device or use `-1` for CPU execution |
| `--results_dir` | Output root; defaults to `results` |
| `--epoch` | Checkpoint suffix; defaults to `latest` |

The script sets the batch size to 1, disables shuffling and flipping, and processes the complete selected split.

## Inference output

Outputs follow this convention:

```text
<results_dir>/<name>/<dataset>_<mode>_translated_<data_mode>/
├── images/
│   ├── <sample>_fake_A.npy
│   └── ...
└── index.html
```

For example, the two test commands above produce:

```text
results/cyclegansen12/SEN12MSCR_test_translated_full/
results/cyclegansen12/SMILECR_test_translated_full/
```

Only the translated `fake_A` result is retained for each sample. Before saving, the inference program normalizes each result to `[0, 1]`, converts it to an HWC `uint8` array in `[0, 255]`, and writes it as NPY.

## Postprocessing generated arrays

The following code reproduces the postprocessing cells in `CycleGAN.ipynb`. Set `folder_name` to the generated result directory and `data_root` to the destination dataset directory.

```python
import os
import shutil
from pathlib import Path

from tqdm import tqdm

# Examples:
#   SEN12MSCR_test_translated_full
#   SEN12MSCR_test_translated_partial
#   SMILECR_test_translated_full
#   SMILECR_test_translated_partial
folder_name = "SEN12MSCR_test_translated_full"

# SEN12MS-CR: the dataset root
data_root = Path("../SEN12MS-CR_1")

# SMILE-CR test alternative:
# data_root = Path("../SMILE-CR-npy/TestData/TestData")

source_dir = Path("results") / "cyclegansen12" / folder_name
images_dir = source_dir / "images"
renamed_dir = source_dir / f"{folder_name}_npy"

if not images_dir.is_dir():
    raise FileNotFoundError(f"Inference output not found: {images_dir}")
if renamed_dir.exists():
    raise FileExistsError(f"Remove or rename the existing directory: {renamed_dir}")

for file_path in tqdm(list(images_dir.iterdir()), desc="Renaming files"):
    if file_path.name.endswith("_fake_A.npy"):
        file_path.rename(file_path.with_name(
            file_path.name.replace("_fake_A.npy", ".npy")
        ))

images_dir.rename(renamed_dir)

destination_dir = data_root / renamed_dir.name
if destination_dir.exists():
    raise FileExistsError(f"Remove or rename the existing directory: {destination_dir}")

shutil.move(str(renamed_dir), str(destination_dir))
print(f"Translated arrays moved to {destination_dir}")
```

The resulting directory names are compatible with the evaluation loaders in the parent SAR-ULRTC-CR project. Typical destinations are:

| Dataset and configuration | Destination |
| --- | --- |
| SEN12MS-CR test, full | `<SEN12MS-CR_1>/SEN12MSCR_test_translated_full_npy/` |
| SEN12MS-CR test, partial | `<SEN12MS-CR_1>/SEN12MSCR_test_translated_partial_npy/` |
| SMILE-CR test, full | `<SMILE-CR-npy>/TestData/TestData/SMILECR_test_translated_full_npy/` |
| SMILE-CR test, partial | `<SMILE-CR-npy>/TestData/TestData/SMILECR_test_translated_partial_npy/` |

Change both `folder_name` and `data_root` for training or validation outputs. For SMILE-CR, place the generated directory under the corresponding `TrainData/TrainData`, `ValData/ValData`, or `TestData/TestData` split directory.

The postprocessing step moves the generated `images/` directory. Preserve a copy first if the original inference layout is needed.

## Troubleshooting

- **Checkpoint not found:** ensure `--name` matches a directory under `checkpoints/` and that the requested `--epoch` files exist.
- **Dataset directory not found:** verify `--dataroot`, `--mode`, `--data_mode`, and the directory names listed above.
- **Unexpected result path:** the output name is constructed from `--dataset`, `--mode`, and `--data_mode`.
- **CUDA device error:** select an available device with `--gpu_ids`, or use `--gpu_ids -1` for CPU inference.
- **Destination already exists:** move or rename the existing translated directory before running the postprocessing code.

## Upstream attribution and citation

This directory is derived from the PyTorch CycleGAN implementation by Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. Please also consult the [upstream documentation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for general CycleGAN usage.

```bibtex
@inproceedings{zhu2017unpaired,
  title     = {Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks},
  author    = {Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A.},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
  year      = {2017}
}
```
