cd /mmfs1/gscratch/krishna/arnabk1/mayank_clip_mrl/src
torchrun --nproc_per_node 4 --master_port 4321 -m training.main \
    --model "ViT-B-16" \
    --train-data "/gscratch/krishna/chenhaoz/IL/FDT/data/cc3m/{00000..00331}.tar"\
    --imagenet-val "/gscratch/krishna/arnabk1/root/val/" \
    --dataset-type webdataset \
    --precision amp \
    --gather-with-grad \
    --local-loss \
    --force_mrl_loss \
    --mrl_loss_weights "1,1,1,1,1" \
    --mrl_dim_to_consider "768,384,192,96,48" \
    --batch-size 256 \
    --train-num-samples 3308333 \
    --accum-freq 32 \
    --workers 4 \
    --epochs 40 \
    --warmup 1000 \
    --zeroshot-frequency 1 \
    --seed 42 \
    --report-to 'wandb' \
    --wandb-project-name "mrl_clip_training" \
    --logs "/mmfs1/gscratch/krishna/arnabk1/mayank_clip_mrl/scripts/logs/" \
    --name "mrl_clip_cc3m_b256_accum_32_gpu_4_ep40_diffLogitScale_D032824_w1" 
