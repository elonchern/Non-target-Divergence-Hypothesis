#!/usr/bin/env bash


# source /home/jsg/jie/pyenv/py3pytorch1.4/bin/activate


CUDA_VISIBLE_DEVICES=0 python ./main.py \
--model='ddrnet' \
--dataset=nyu \
--epochs=5 \
--batch_size=4 \
--workers=4 \
--lr=0.01 \
--lr_adj_n=1 \
--lr_adj_rate=0.1 \
--model_name='SSC_DDRNet' 2>&1 |tee train_DDRNet_NYUCAD.log


# deactivate

