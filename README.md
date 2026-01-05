# PRISM: Diversifying Dataset Distillation by Decoupling Architectural Priors

Official PyTorch implementation of paper (TMLR'26) PRISM : Diversifying Dataset Distillation by Decoupling Architectural Priors


## Abstract

Dataset distillation (DD) promises compact yet faithful synthetic data, but existing approaches often inherit the inductive bias of a single teacher model. As dataset size increases, this bias drives generation toward overly smooth, homogeneous samples, reducing intra-class diversity and limiting generalization. We present PRISM (PRIors from diverse Source Models), a framework that disentangles architectural priors during synthesis. PRISM decouples the logit-matching and regularization objectives, supervising them with different teacher architectures: a primary model for logits and a stochastic subset for batch-normalization (BN) alignment. On ImageNet-1K, PRISM consistently and reproducibly outperforms single-teacher methods (e.g., SRe2L) and recent multi-teacher variants (e.g., G-VBSM) at low- and mid-IPC regimes. The generated data also show significantly richer intra-class diversity, as reflected by a notable drop in cosine similarity between features. We further analyze teacher selection strategies (pre- vs. intra-distillation) and introduce a scalable cross-class batch formation scheme for fast parallel synthesis. Code will be released after the review period.


## Recover 

More details in [recover/README.md](recover/README.md).

We provide scripts for different ResNet architectures:
- `recover/recover_r18.sh` - ResNet-18
- `recover/recover_r50.sh` - ResNet-50
- `recover/recover_r101.sh` - ResNet-101

```bash
cd recover
sh recover_r18.sh  # or recover_r50.sh, recover_r101.sh
```

## Relabel

More details in [relabel/README.md](relabel/README.md).

We provide scripts for different ResNet architectures:
- `relabel/relabel_r18.sh` - ResNet-18
- `relabel/relabel_r50.sh` - ResNet-50
- `relabel/relabel_r101.sh` - ResNet-101

```bash
cd relabel
sh relabel_r18.sh  # or relabel_r50.sh, relabel_r101.sh
```

## Validate

More details in [validate/README.md](validate/README.md).

We provide scripts for different ResNet architectures:
- `validate/train_FKD_10_r18.sh` - ResNet-18
- `validate/train_FKD_10_r50.sh` - ResNet-50
- `validate/train_FKD_10_r101.sh` - ResNet-101

```bash
cd validate
sh train_FKD_10_r18.sh  # or train_FKD_10_r50.sh, train_FKD_10_r101.sh
```

## Citation

If you find our code useful for your research, please cite our paper.

```
@article{moser2025prism,
  title={PRISM: Diversifying Dataset Distillation by Decoupling Architectural Priors},
  author={Moser, Brian B and Sarode, Shalini and Raue, Federico and Frolov, Stanislav and Adamkiewicz, Krzysztof and Shanbhag, Arundhati and Folz, Joachim and Nauen, Tobias C and Dengel, Andreas},
  journal={arXiv preprint arXiv:2511.09905},
  year={2025}
}
```

