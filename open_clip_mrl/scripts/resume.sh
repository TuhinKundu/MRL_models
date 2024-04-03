cd /mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src
torchrun --nproc_per_node 2 -m --master_port 3436 training.main \
    --model "ViT-B-16" \
    --train-data "/mmfs1/data/yfcc-tmp/cc_3m/train_shards/shard_{000000..003318}.tar" \
    --imagenet-val "/mmfs1/data/yfcc-tmp/imagenet/val/" \
    --dataset-type webdataset \
    --precision amp \
    --gather-with-grad \
    --local-loss \
    --accum-freq 1 \
    --workers 4 \
    --epochs 60 \
    --warmup 1000 \
    --zeroshot-frequency 1 \
    --seed 0 \
    --resume "/mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src/logs/mrl_clip/clip_b512_accum_1_ep40_bugfixed/checkpoints/epoch_40.pt" \
    --report-to 'wandb' \
    --wandb-project-name "mrl_clip_training" \
    --logs "/mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src/logs/mrl_clip" \
    --name "clip_b512_accum_1_ep40_resumefrom40" 


    # --force_mrl_loss \
    # --mrl_loss_weights "1,1,1,1,1" \
    # --mrl_dim_to_consider "768,384,192,96,48" \