#!/bin/bash
#SBATCH -p gpu-large
###SBATCH -w oven-0-13
#SBATCH --gres gpu:1
#SBATCH -c 16
#SBATCH --mem=200Gb
#SBATCH --job-name location_zhuang1
#SBATCH --output /work/magroup/skrieger/tissue_generator/spencer_mimyr/models/slurm/logs/location_zhuang1-%J.txt
#SBATCH -t 1-00:00:00

module load cuda-12.1
source /work/magroup/skrieger/newconda/etc/profile.d/conda.sh
conda activate scMulan2-slaf
cd /work/magroup/skrieger/tissue_generator/spencer_mimyr
mkdir -p /scratch/skrieger model_checkpoints
export WANDB_DIR=/scratch/skrieger

# Retrain the location DDPM on Zhuang-1 (data_mode base_zhuang1), the same slices
# the expression + cell-type modules were trained on. The bisection showed location
# is the dominant bottleneck (~0.10 soft_spearman, ~2x the cell-type cost). Feeds
# aligned_spatial ([z_ccf, y_ccf, x_ccf]) + cond=[4,5,x_ccf,0,0,1] to DDPMTrainer.train().
# Saves to model_checkpoints/location_zhuang1_ddpm_2d_checkpoint_<epoch>.pt (does NOT
# clobber the production smoothtune_..._400.pt).
python train_location_zhuang1.py \
  --epochs 400 \
  --batch_size 4096 \
  --ckpt_prefix location_zhuang1_ddpm_2d_checkpoint

echo "===== done; checkpoints at model_checkpoints/location_zhuang1_ddpm_2d_checkpoint_<epoch>.pt ====="
