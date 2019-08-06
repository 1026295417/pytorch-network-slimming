#!/bin/bash


# prune 30% channels
python train.py --arch densenet63 \
                --prune-ratio 0.3 \
                --resume-path output-densenet63-bn-sparsity/ckpt_best.pth \
                --epochs 450 --lr-decay-epochs 100 \
                --tfs \
                --outf output-densenet63-bn-pr03-tfs

# prune 50% channels
python train.py --arch densenet63 \
                --prune-ratio 0.5 \
                --resume-path output-densenet63-bn-sparsity/ckpt_best.pth \
                --epochs 450 --lr-decay-epochs 100 \
                --tfs \
                --outf output-densenet63-bn-pr05-tfs

# prune 70% channels
python train.py --arch densenet63 \
                --prune-ratio 0.7 \
                --resume-path output-densenet63-bn-sparsity/ckpt_best.pth \
                --epochs 450 --lr-decay-epochs 100 \
                --tfs \
                --outf output-densenet63-bn-pr07-tfs
