#!/bin/bash

### Sparsity on prunable BNs only ###
# prune 30% channels
python train.py --arch resnet18 \
                --prune-ratio 0.3 \
                --resume-path output-resnet18-bn-sparsity/ckpt_best.pth \
                --epochs 450 --lr-decay-epochs 100 \
                --tfs \
                --outf output-resnet18-bn-pr03-tfs

# prune 50% channels
python train.py --arch resnet18 \
                --prune-ratio 0.5 \
                --resume-path output-resnet18-bn-sparsity/ckpt_best.pth \
                --epochs 450 --lr-decay-epochs 100 \
                --tfs \
                --outf output-resnet18-bn-pr05-tfs

# prune 70% channels
python train.py --arch resnet18 \
                --prune-ratio 0.7 \
                --resume-path output-resnet18-bn-sparsity/ckpt_best.pth \
                --epochs 450 --lr-decay-epochs 100 \
                --tfs \
                --outf output-resnet18-bn-pr07-tfs

### Sparsity on all BNs ###
# prune 30% channels
python train.py --arch resnet18 \
                --prune-ratio 0.3 \
                --resume-path output-resnet18-all-bn-sparsity/ckpt_best.pth \
                --epochs 450 --lr-decay-epochs 100 \
                --tfs \
                --outf output-resnet18-all-bn-pr03-tfs

# prune 50% channels
python train.py --arch resnet18 \
                --prune-ratio 0.5 \
                --resume-path output-resnet18-all-bn-sparsity/ckpt_best.pth \
                --epochs 450 --lr-decay-epochs 100 \
                --tfs \
                --outf output-resnet18-all-bn-pr05-tfs

# prune 70% channels
python train.py --arch resnet18 \
                --prune-ratio 0.7 \
                --resume-path output-resnet18-all-bn-sparsity/ckpt_best.pth \
                --epochs 450 --lr-decay-epochs 100 \
                --tfs \
                --outf output-resnet18-all-bn-pr07-tfs
