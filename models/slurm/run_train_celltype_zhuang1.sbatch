#!/bin/bash
#SBATCH -p gpu-large
###SBATCH -w oven-0-13
#SBATCH --gres gpu:1
#SBATCH -c 16
#SBATCH --mem=200Gb
#SBATCH --job-name celltype_zhuang1
#SBATCH --output /work/magroup/skrieger/tissue_generator/spencer_mimyr/models/slurm/logs/celltype_zhuang1-%J.txt
#SBATCH -t 1-00:00:00

module load cuda-12.1
source /work/magroup/skrieger/newconda/etc/profile.d/conda.sh
conda activate scMulan2-slaf
cd /work/magroup/skrieger/tissue_generator/spencer_mimyr
mkdir -p /scratch/skrieger model_checkpoints
export WANDB_DIR=/scratch/skrieger

# Retrain ONLY the cell-type module on Zhuang-1 (data_mode base_zhuang1), the same
# slices the expression module was trained on. Loader cfg matches the rq3_v2
# distance eval so the token vocab is consistent. Saves to
# model_checkpoints/celltype_zhuang1.pt. Location DDPM retrain is deferred.
python train_celltype_zhuang1.py \
  --out model_checkpoints/celltype_zhuang1.pt \
  --epochs 1000 \
  --batch_size 1024

echo "===== done; checkpoint at model_checkpoints/celltype_zhuang1.pt ====="
