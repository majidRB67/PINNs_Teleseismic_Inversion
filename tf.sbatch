#!/bin/bash                                                                             
#SBATCH -t 24:00:00
#SBATCH --mem=10g
#SBATCH --partition=gpu-he --gres=gpu:1
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -o tf_job%j.out
#SBATCH -e tf_job%j.err


module load tensorflow/1.14.0_gpu_py36

module load python/3.6.6_test

module load cuda/10.0.130

module load cudnn/7.4

export PYTHONUNBUFFERED=TRUE

python3 inv_acous_multiEvent_batch.py