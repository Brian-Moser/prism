# Generate & Save Soft Labels from Supervised Teachers

## Preparation

- Python >= 3.8
- PyTorch >= 2.0.0
- Torchvision >= 0.15.1

### Additional Dependencies

Install the required packages:
```bash
pip install timm
```

## Soft Labels Generation

To relabel distilled data, we provide scripts for different ResNet architectures:
- `relabel_r18.sh` - ResNet-18
- `relabel_r50.sh` - ResNet-50
- `relabel_r101.sh` - ResNet-101

Example usage with ResNet-18:
```bash
python generate_soft_label_r18.py \
    -b 50 \
    -j 8 \
    --epochs 300 \
    --fkd-seed 42 \
    --input-size 224 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1 \
    --use-fp16 \
    --candidate-number 4 \
    --fkd-path FKD_prism_r18 \
    --mode 'fkd_save' \
    --mix-type 'cutmix' \
    --data /path/to/recover/syn_data/full_r18_10
```

Or simply run the provided script:
```bash
sh relabel_r18.sh  # or relabel_r50.sh, relabel_r101.sh
```

Note: The scripts use architecture-specific Python files:
- `generate_soft_label_r18.py` for ResNet-18
- `generate_soft_label_r50.py` for ResNet-50
- `generate_soft_label_r101.py` for ResNet-101

## Make FKD Compatible with Mixup and CutMix

As illustrated below, we modify the [`FKD`](https://github.com/szq0214/FKD) code to make it compatible with `Mixup` and `CutMix`. In detail, `Crop Coords` in `RandomResizedCrop` operation, `Flip Status` in `RandomHorizontalFlip` operation, and `Mixup index, ratio, bbox` in `Mixup/CutMix` augmentation, and soft label are saved as configuration files. The saved configuration files will be loaded in [training on relabeled data](../validate).

<div align=left>
<img style="width:70%" src="../img/fkd-mix.png">
</div>

## Usage

```
usage: generate_soft_label_r18.py (or generate_soft_label_r50.py, generate_soft_label_r101.py)

[-h] [--data DIR] [-a ARCH] [-j N] [-b N] [--world-size WORLD_SIZE] [--rank RANK] [--dist-url DIST_URL] [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU] [--multiprocessing-distributed] [--epochs EPOCHS] [--input-size S] [--min-scale-crops MIN_SCALE_CROPS] [--max-scale-crops MAX_SCALE_CROPS] [--fkd-path FKD_PATH] [--use-fp16] [--candidate-number CANDIDATE_NUMBER] [--mode N] [--fkd-seed N] [--mix-type {mixup,cutmix,None}] [--mixup MIXUP] [--cutmix CUTMIX]

arguments:
  -h, --help            show this help message and exit
  --data DIR            path to dataset
  -a ARCH, --arch ARCH
  -j N, --workers N     number of data loading workers (default: 4)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training
  --epochs EPOCHS
  --input-size S        argument in RandomResizedCrop
  --min-scale-crops MIN_SCALE_CROPS
                        argument in RandomResizedCrop
  --max-scale-crops MAX_SCALE_CROPS
                        argument in RandomResizedCrop
  --fkd-path FKD_PATH   path to save soft labels
  --use-fp16            save soft labels as `fp16`
  --candidate-number CANDIDATE_NUMBER
                        number of candidate models for ensemble
  --mode N
  --fkd-seed N
  --mix-type {mixup,cutmix,None}
                        mixup or cutmix or None
  --mixup MIXUP         mixup alpha, mixup enabled if > 0. (default: 0.8)
  --cutmix CUTMIX       cutmix alpha, cutmix enabled if > 0. (default: 1.0)
```




## Download soft labels from [![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/zeyuanyin/SRe2L)
| dataset | resolution | iteration | IPC | files |
|:---:|:---:|:---:|:---:| :---:|
| ImageNet-1K | 224x224 | 4K | 50 | [images](https://huggingface.co/datasets/zeyuanyin/SRe2L/resolve/main/sre2l_in1k_rn18_4k_ipc50.zip) <br> [mixup labels](https://huggingface.co/datasets/zeyuanyin/SRe2L/resolve/main/FKD_mixup_fp16.tar.gz) / [cutmix labels](https://huggingface.co/datasets/zeyuanyin/SRe2L/resolve/main/FKD_cutmix_fp16.tar.gz) |