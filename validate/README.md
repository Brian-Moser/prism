# Validate Performance of Distilled Data

## Preparation

- Python >= 3.8
- PyTorch >= 2.0.0
- Torchvision >= 0.15.1

### Additional Dependencies

Install the required packages:
```bash
pip install wandb
pip install --upgrade typing_extensions
pip install timm
pip install prefetch_generator
```

After installation, configure wandb:
```bash
wandb enabled
wandb online
```

- Modify PyTorch source code `torch.utils.data._utils.fetch._MapDatasetFetcher` to support *multi-processing loading* of soft label data and mix configurations.
  ```python
  ### Original code
  class _MapDatasetFetcher(_BaseDatasetFetcher):
      def fetch(self, possibly_batched_index):
          if self.auto_collation:
              if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                  data = self.dataset.__getitems__(possibly_batched_index)
              else:
                  data = [self.dataset[idx] for idx in possibly_batched_index]
          else:
              data = self.dataset[possibly_batched_index]
          return self.collate_fn(data)

  ### Modified code
  class _MapDatasetFetcher(_BaseDatasetFetcher):
      def fetch(self, possibly_batched_index):
          if hasattr(self.dataset, "mode") and self.dataset.mode == 'fkd_load':
              mix_index, mix_lam, mix_bbox, soft_label = self.dataset.load_batch_config(possibly_batched_index[0])

          if self.auto_collation:
              if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                  data = self.dataset.__getitems__(possibly_batched_index)
              else:
                  data = [self.dataset[idx] for idx in possibly_batched_index]
          else:
              data = self.dataset[possibly_batched_index]

          if hasattr(self.dataset, "mode") and self.dataset.mode == 'fkd_load':
              return self.collate_fn(data), mix_index.cpu(), mix_lam, mix_bbox, soft_label.cpu()
          else:
              return self.collate_fn(data)
  ```

## Training on Relabeled Distilled Data

To train a model on relabeled distilled data, we provide scripts for different ResNet architectures:
- `train_FKD_10_r18.sh` - ResNet-18
- `train_FKD_10_r50.sh` - ResNet-50
- `train_FKD_10_r101.sh` - ResNet-101

Example usage with ResNet-18:
```bash
python train_FKD_parallel.py \
    --wandb-project 'val_rn18_imnet_10' \
    --batch-size 50 \
    --gradient-accumulation-steps 2 \
    --model resnet18 \
    --cos \
    --loss-type "mse_gt" \
    --ce-weight 0.1 \
    -j 4 \
    -T 20 \
    --mix-type 'cutmix' \
    --output-dir ./save/val_rn18_imnet_10/ \
    --train-dir /path/to/recover/syn_data/full_r18/ \
    --val-dir /path/to/imagenet/val/ \
    --fkd-path /path/to/relabel/FKD_prism_r18/
```

Or simply run the provided script:
```bash
sh train_FKD_10_r18.sh  # or train_FKD_10_r50.sh, train_FKD_10_r101.sh
```

Since we modify the PyTorch source code to load the soft labels data and mix configurations before fetching batch data, it will take more memory than the original code due to extra files to be temporarily stored in memory. Thus, we recommend to use a smaller `-j` number of workers to load data and use a larger `--gradient-accumulation-steps` to reduce the memory in model inference. For reference, we use `-j 4 --gradient-accumulation-steps 4` in single RTX 4090 with 24GB memory, `-j 8 --gradient-accumulation-steps 1` in single Tesla A100 with 40GB. There is no effect on `val_loader`, whose `num_workers` can be set to a larger number.

In terms of the FKD-related arguments, they should align to the setting in [relabel](../relabel). For example, `--batch-size` should be the same value in [relabel](../relabel) and `--epochs` argument should be no more than the epochs in [relabel](../relabel).

In terms of the `--gradient-accumulation-steps` argument, it will split the loaded batch data of `--batch-size` into some smaller batch data. For example, if `--gradient-accumulation-steps 4`, it will split the loaded batch data of 1024 into 4 smaller batch data of 256 each. Then, it will accumulate the gradients of 4 smaller batch data and update the model parameters once. In this way, it can reduce the memory in model inference.

In terms of `wandb`, we use it to record the training process. If you don't want to use it, you can set `wandb disabled` in `train.sh`. If you want to use it, you need to set `wandb enabled \\ wandb online` and `--wandb-api-key` in `train.sh`.

## Alternative Validation

We provide an alternative validation code with naive KD to quickly validate the performance of the distilled data without the relabel process. The usage is similar to the FKD training code `train_FKD_parallel.py`. Note that the naive KD code may not be available in this repository version.

## Usage

```
usage: train_FKD_parallel.py

[-h] [--batch-size BATCH_SIZE] [--gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS] [--start-epoch START_EPOCH] [--epochs EPOCHS] [-j WORKERS] [--train-dir TRAIN_DIR] [--val-dir VAL_DIR] [--output-dir OUTPUT_DIR] [--cos] [--sgd] [-lr LEARNING_RATE] [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY] [--adamw-lr ADAMW_LR] [--adamw-weight-decay ADAMW_WEIGHT_DECAY] [--model MODEL] [--keep-topk KEEP_TOPK] [-T TEMPERATURE] [--fkd-path FKD_PATH] [--wandb-project WANDB_PROJECT] [--loss-type LOSS_TYPE] [--ce-weight CE_WEIGHT] [--mix-type {mixup,cutmix,None}] [--fkd_seed FKD_SEED]

arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size
  --gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS
                        gradient accumulation steps for small gpu memory
  --start-epoch START_EPOCH
                        start epoch
  --epochs EPOCHS       total epoch
  -j WORKERS, --workers WORKERS
                        number of data loading workers
  --train-dir TRAIN_DIR
                        path to training dataset
  --val-dir VAL_DIR     path to validation dataset
  --output-dir OUTPUT_DIR
                        path to output dir
  --cos                 cosine lr scheduler
  --sgd                 sgd optimizer
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        sgd init learning rate
  --momentum MOMENTUM   sgd momentum
  --weight-decay WEIGHT_DECAY
                        sgd weight decay
  --adamw-lr ADAMW_LR   adamw learning rate
  --adamw-weight-decay ADAMW_WEIGHT_DECAY
                        adamw weight decay
  --model MODEL         student model name
  --keep-topk KEEP_TOPK
                        keep topk logits for kd loss
  -T TEMPERATURE, --temperature TEMPERATURE
                        temperature for distillation loss
  --fkd-path FKD_PATH   path to fkd label
  --wandb-project WANDB_PROJECT
                        wandb project name
  --wandb-api-key WANDB_API_KEY
                        wandb api key
  --loss-type LOSS_TYPE
                        loss type for training (e.g., "mse_gt")
  --ce-weight CE_WEIGHT
                        weight for cross-entropy loss
  --mix-type {mixup,cutmix,None}
                        mixup or cutmix or None
  --fkd_seed FKD_SEED   seed for batch loading sampler
```


