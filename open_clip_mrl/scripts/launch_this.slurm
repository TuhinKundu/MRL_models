#!/bin/bash

#SBATCH --account=krishna
#SBATCH --partition=gpu-a40
#SBATCH --job-name=MClip2_cc3m_mrl
#SBATCH --error=./jobs/job.err.%j  
#SBATCH --output=./jobs/job.run.%j
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=4
#SBATCH --mem-per-gpu=40G
#SBATCH --nodes=1
#SBATCH --mail-user=arnabk1@uw.edu
#SBATCH --mail-type=ALL

bash finetuning_mrl.sh
