#!/bin/bash
#SBATCH -p gpu-large
###SBATCH -w oven-0-13
#SBATCH --gres gpu:1
#SBATCH -c 16
#SBATCH --mem=200Gb
#SBATCH --job-name celltype_nbhd
#SBATCH --output /work/magroup/skrieger/tissue_generator/spencer_mimyr/models/slurm/logs/celltype_nbhd-%J.txt
#SBATCH -t 1-00:00:00

module load cuda-12.1
source /work/magroup/skrieger/newconda/etc/profile.d/conda.sh
conda activate scMulan2-slaf
cd /work/magroup/skrieger/tissue_generator/spencer_mimyr
mkdir -p /scratch/skrieger model_checkpoints
export WANDB_DIR=/scratch/skrieger

# Neighborhood-KL cell-type model: same Zhuang-1 data / regime as celltype_zhuang1.pt
# (epochs, batch, patience, de-noised early stop) but the loss fits each cell's
# softmax to the cell-type DISTRIBUTION of its 20-NN neighborhood (forward KL /
# soft-CE), instead of the one-hot label. Goal: make rare types concentrate where
# GT does (fix the uniform ~4% base rate the pairwise probe exposed) rather than
# being spatially flat. lambda=1.0 = pure neighborhood distribution.
# Saves to celltype_zhuang1_nbhd.pt (does NOT clobber celltype_zhuang1.pt).
python train_celltype_zhuang1.py \
  --out model_checkpoints/celltype_zhuang1_nbhd.pt \
  --epochs 1000 \
  --batch_size 1024 \
  --patience 20 \
  --neighborhood_k 20 \
  --neighborhood_weight 1.0 \
  --neighbor_cache model_checkpoints/nbrtokens_zhuang1_k20.npy

echo "===== done; checkpoint at model_checkpoints/celltype_zhuang1_nbhd.pt ====="
