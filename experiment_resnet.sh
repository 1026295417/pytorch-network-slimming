#!/bin/bash

# baseline
python train.py --arch resnet18 --outf output-resnet18

### Sparsity on prunable BNs only ###
python train.py --arch resnet18 --l1-decay 2e-4 \
                --outf output-resnet18-bn-sparsity

# prune 30% channels
python train.py --arch resnet18 \
                --prune-ratio 0.3 \
                --resume-path output-resnet18-bn-sparsity/ckpt_best.pth \
                --epochs 30 --lr-decay-epochs 8 --lr-decay-factor 0.1 --lr 0.001 \
                --weight-decay 2e-2 \
                --outf output-resnet18-bn-pr03

# prune 50% channels
python train.py --arch resnet18 \
                --prune-ratio 0.5 \
                --resume-path output-resnet18-bn-sparsity/ckpt_best.pth \
                --epochs 50 --lr-decay-epochs 14 --lr-decay-factor 0.1 --lr 0.001 \
                --weight-decay 3e-2 \
                --outf output-resnet18-bn-pr05

# prune 70% channels
python train.py --arch resnet18 \
                --prune-ratio 0.7 \
                --resume-path output-resnet18-bn-sparsity/ckpt_best.pth \
                --epochs 100 --lr-decay-epochs 30 --lr-decay-factor 0.1 --lr 0.001 \
                --weight-decay 7e-3 \
                --outf output-resnet18-bn-pr07

### Sparsity on all BNs ###
python train.py --arch resnet18 --l1-decay 2e-4 \
                --outf output-resnet18-all-bn-sparsity

# prune 30% channels
python train.py --arch resnet18 \
                --prune-ratio 0.3 \
                --resume-path output-resnet18-all-bn-sparsity/ckpt_best.pth \
                --epochs 30 --lr-decay-epochs 8 --lr-decay-factor 0.1 --lr 0.001 \
                --weight-decay 2e-2 \
                --outf output-resnet18-all-bn-pr03

# prune 50% channels
python train.py --arch resnet18 \
                --prune-ratio 0.5 \
                --resume-path output-resnet18-all-bn-sparsity/ckpt_best.pth \
                --epochs 50 --lr-decay-epochs 14 --lr-decay-factor 0.1 --lr 0.001 \
                --weight-decay 3e-2 \
                --outf output-resnet18-all-bn-pr05

# prune 70% channels
python train.py --arch resnet18 \
                --prune-ratio 0.7 \
                --resume-path output-resnet18-all-bn-sparsity/ckpt_best.pth \
                --epochs 100 --lr-decay-epochs 30 --lr-decay-factor 0.1 --lr 0.001 \
                --weight-decay 7e-3 \
                --outf output-resnet18-all-bn-pr07
