#!/bin/bash

# prune 30% channels
python train.py --arch vgg11s \
                --prune-ratio 0.3 \
                --resume-path output-vgg11s-bn-sparsity/ckpt_best.pth \
                --epochs 450 --lr-decay-epochs 100 \
                --tfs \
                --outf output-vgg11s-bn-pr03-tfs

# prune 50% channels
python train.py --arch vgg11s \
                --prune-ratio 0.5 \
                --resume-path output-vgg11s-bn-sparsity/ckpt_best.pth \
                --epochs 450 --lr-decay-epochs 100 \
                --tfs \
                --outf output-vgg11s-bn-pr05-tfs

# prune 70% channels
python train.py --arch vgg11s \
                --prune-ratio 0.7 \
                --resume-path output-vgg11s-bn-sparsity/ckpt_best.pth \
                --epochs 450 --lr-decay-epochs 100 \
                --tfs \
                --outf output-vgg11s-bn-pr07-tfs
