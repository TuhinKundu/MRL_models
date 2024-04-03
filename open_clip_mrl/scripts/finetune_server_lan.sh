#!/bin/bash

accelerate launch --num_processes 4 --config_file ../../clip_model/default_config.yaml training/main.py \
--model ViT-B-16 \
--pretrained laion400m_e32 \
--train-data "/tk/clip/cc12m/{00000..01242}.tar" \
--imagenet-val /tk/imagenet/val \
--dataset-type webdataset \
--precision bf16 \
--gather-with-grad \
--local-loss \
--force_mrl_loss \
--mrl_loss_weights 1,1,1,1 \
--mrl_dim_to_consider 768,384,192,96 \
--accum-freq 32 \
--batch-size 1024 \
--train-num-samples 6029899 \
--lr 2e-05 \
--workers 8 \
--epochs 5 \
--warmup 500 \
--zeroshot-frequency 1 \
--seed 42 \
--report-to 'wandb' \
--wandb-project-name mrl_clip_training \
--logs logs/ \
--wandb_key 08cc460d0b9627616da48f73cff12909aecffd83 \
--lr_scheduler cosine