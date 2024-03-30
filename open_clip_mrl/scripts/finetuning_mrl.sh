cd /mmfs1/gscratch/krishna/arnabk1/open_clip_mrl/src
torchrun --nproc_per_node 4 --master_port 4329 -m training.main \
    --model "ViT-B-16" \
    --pretrained "laion400m_e32" \
    --train-data "/gscratch/krishna/chenhaoz/IL/FDT/data/cc3m/{00000..00331}.tar" \
    --imagenet-val "/gscratch/krishna/arnabk1/root/val/" \
    --dataset-type webdataset \
    --precision amp \
    --gather-with-grad \
    --local-loss \
    --force_mrl_loss \
    --mrl_loss_weights "1,1,1,1,1" \
    --mrl_dim_to_consider "768,384,192,96,48" \
    --accum-freq 32 \
    --batch-size 256 \
    --train-num-samples 3308333 \
    --lr 1e-07 \
    --workers 4 \
    --epochs 10 \
    --warmup 500 \
    --zeroshot-frequency 1 \
    --seed 42 \
    --report-to 'wandb' \
    --wandb-project-name "mrl_clip_training" \
    --logs "/gscratch/krishna/arnabk1/mayank_clip_mrl/scripts/logs/" \
    --name "ViT-B-16_liaon400m_e32_b256_accum32_gpu4_finetune_mrl_ep10_warmup_500_lr1e-07_"

