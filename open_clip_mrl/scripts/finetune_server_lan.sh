#!/bin/bash

accelerate launch --num_processes 2 --config_file ../../clip_model/default_config.yaml training/main.py \
--model ViT-B-16 \
--pretrained laion400m_e32 \
--train-data "data/cc3m/{00000..00331}.tar" \
--imagenet-val /home/tuhin/uw/CLIP_benchmark/root/val \
--dataset-type webdataset \
--precision bf16 \
--gather-with-grad \
--local-loss \
--force_mrl_loss \
--mrl_loss_weights 1,1,1,1,1 \
--mrl_dim_to_consider 768,384,192,96,48 \
--accum-freq 2 \
--batch-size 64 \
--train-num-samples 2701900 \
--lr 1e-07 \
--workers 4 \
--epochs 10 \
--warmup 500 \
--zeroshot-frequency 1 \
--seed 42 \
--report-to 'wandb' \
--wandb-project-name mrl_clip_training \
--logs logs/ \
--name ViT-B-16_liaon400m_e32_b256_accum32_gpu4_finetune_mrl_ep10_warmup_500_lr1e-07_ \
--wandb_key 08cc460d0b9627616da48f73cff12909aecffd83 \
--lr_scheduler cosine