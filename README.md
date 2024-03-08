### Text and multimodal pretraining

This repository uses open source tools provided below to run models and corresponding versions. 
These libraries support wide variants of hardware, parallelism, training optimizations and logging.
Default config provided here is for Nvidia GPUs with Ampere architecture and above (e.g A100s, RTX-30*/40* or later) 
for utilizing bfloat16 compute optimization paired with deepspeed zero2 for optimizer partitioning that allows 
high batch sizes due to lower memory footprint.
Please install the following: 

##### Dependencies
Codebase developed with 
    
    torch==2.0.1+cu117
    transformers==4.37.2
    deepspeed==0.13.2
    accelerate==0.27.2
    open_clip_torch==2.24.0
    wandb

Please configure HF Accelerate library by running `accelerate config` to utilize appropriate hardware. In the following 
questions asked, select deepspeed with appropriate hardware configuration and provide pathname for the deepspeed 
config file [similar to this](https://github.com/TuhinKundu/MRL_models/blob/main/bert/zero2_config_accelerate.json). The 
code uses HF Accelerate like a wrapper around deepspeed backbone to provide wider range of support to models in 
HF Transformers and Open CLIP libraries.

#### MLM pretraining

    accelerate launch --config_file bert/default_config.yaml run_pretraining.py --batch_size 2 --mrl yes --clip n --mlm y  --data_path ../bookcorpus_train/ --output_path output_dir/ --model_name bert-base-uncased 

##### Open_Clip pretraining

Open CLIP pretraining has been integrated here with HF Accelerate (with deepspeed integration)
pipeline to support both HF and Open CLIP models with deepspeed zero1/2 stage training paradigm.

Please download the appropriate dataset from [Datacomp](https://github.com/mlfoundations/datacomp) or [img2dataset](https://github.com/rom1504/img2dataset)
github repositories. To play around, download [cc3m](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md) dataset that takes
less than an hour to download.

Example command to run clip pretraining

    accelerate launch --config_file clip/default_config.yaml run_pretraining.py --batch_size 8 --mrl yes --clip yes --data_path clip/cc3m/ --output_path output_dir/ --model_name ViT-B-32 --evaluation_interval 100

Miscellaneous

1. Check model name with `open_clip.list_models()` to pass argument for `model_name`. 
Please check list of model names inside config folder in [open_clip](https://github.com/mlfoundations/open_clip/tree/main/src/open_clip/model_configs)
github repository.You can also pass any custom model config supported by `open_clip`.
2. In MLM, if you mention `train_batch_size` or `train_micro_batch_size_per_gpu` parameters in deepspeed config,
it will overwrite the `global_batch_size` or `batch_size` parameters you pass as arguments. 
In CLIP, we use OpenCLIP's dataloading strategy hence you **need** to pass either `global_batch_size` or `batch_size` parameters
as the dataloader will ignore deepspeed parameters. Please use default `auto` parameter in deepspeed config for CLIP.