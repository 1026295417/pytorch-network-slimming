#!/bin/bash

# baseline
python train.py --arch densenet63 --outf output-densenet63

# baseline with sparsity on BNs
python train.py --arch densenet63 --l1-decay 1e-4 \
                --outf output-densenet63-bn-sparsity

# prune 30% channels
python train.py --arch densenet63 \
                --prune-ratio 0.3 \
                --resume-path output-densenet63-bn-sparsity/ckpt_best.pth \
                --epochs 30 --lr-decay-epochs 8 --lr-decay-factor 0.1 --lr 0.001 \
                --weight-decay 1e-2 \
                --outf output-densenet63-bn-pr03

# prune 50% channels
python train.py --arch densenet63 \
                --prune-ratio 0.5 \
                --resume-path output-densenet63-bn-sparsity/ckpt_best.pth \
                --epochs 50 --lr-decay-epochs 14 --lr-decay-factor 0.1 --lr 0.001 \
                --weight-decay 2e-3 \
                --outf output-densenet63-bn-pr05

# prune 50% channels
python train.py --arch densenet63 \
                --prune-ratio 0.7 \
                --resume-path output-densenet63-bn-sparsity/ckpt_best.pth \
                --epochs 100 --lr-decay-epochs 30 --lr-decay-factor 0.1 --lr 0.001 \
                --weight-decay 2e-3 \
                --outf output-densenet63-bn-pr07
