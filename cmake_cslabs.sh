#!/bin/bash
#SBATCH --job-name=build_fa
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=0:05:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# srun --partition=gpunodes --gres=gpu:rtx_4090:1 -t 5 cmake.sh

# while [ ! -z "$(dcgmi -v | grep 'Hostengine build info:')" ]; do
#   sleep 5;
# done
# export PATH=/u/wangke61/.local/cuda-12.9/bin:$PATH
# export LD_LIBRARY_PATH=/u/wangke61/.local/cuda-12.9/lib64:$LD_LIBRARY_PATH

mkdir -p build
cd build

nvidia-smi -L

# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

cc=86
cmake -S .. -B . -DCC=${cc}

make

# ncu --set full -o kernel -f --import-source on (--launch-count 1) for nsight compute

echo -e "Running ThunderKittens\n\n\n"
./fa4090TK ../randn_1024_64.txt
echo -e "Running CUDA\n\n\n"
./fa_ampere_cuda ../randn_1024_64.txt
echo -e "Running CUBLAS GEMM"
./gemm