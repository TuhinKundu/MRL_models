cd /mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src
torchrun --nproc_per_node 2 --master_port 3467 -m training.main \
    --model "ViT-B-16" \
    --train-data "/mmfs1/data/yfcc-tmp/cc_3m/train_shards/shard_{000000..003318}.tar" \
    --imagenet-val "/mmfs1/data/yfcc-tmp/imagenet/val/" \
    --dataset-type webdataset \
    --precision amp \
    --gather-with-grad \
    --local-loss \
    --batch-size 512 \
    --accum-freq 1 \
    --workers 4 \
    --epochs 40 \
    --warmup 4000 \
    --zeroshot-frequency 2 \
    --seed 0 \
    --report-to 'wandb' \
    --wandb-project-name "mrl_clip_training" \
    --logs "/mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src/logs/mrl_clip" \
    --name "clip_b512_accum_1_ep40_bugfixed" 