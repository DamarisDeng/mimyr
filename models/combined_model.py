from dataclasses import dataclass
from models.diffusion_model import DDPMTrainer
from models.celltype_model import SkeletonCelltypeModel2
from data_loader import _read_slice
import torch

@dataclass
class TrainConfig:
    degree: int = 7
    hidden_sizes: tuple = (1024, 2048, 4096, 2048, 1024)
    activation: str = "silu"
    batchnorm: bool = False
    dropout: float = 0.0
    feature_type: str = "poly"
    num_rff_features: int = 256
    rff_gamma: float = 100.0
    rff_seed: int | None = None
    n_timesteps: int = 70
    schedule_type: str = "cosine"
    beta_start: float = 1e-10
    beta_end: float = 1e-9
    cosine_s: float = 0.008
    batch_size: int = 4096 * 50
    lr: float = 2e-4
    weight_decay: float = 0
    epochs: int = 1000000
    grad_clip: float = None
    ema_decay: float = 0.999
    # checkpoint basename prefix (epoch appended). Override for a retrain so it
    # doesn't clobber the production smoothtune_..._400.pt checkpoint.
    ckpt_prefix: str = "smoothtune2_conditional_ddpm_2d_checkpoint"
    # whether train() renders the in-loop 200k-point sample figures (headless: off).
    viz: bool = True

class CombinedModel:
    def __init__(self, location_model_checkpoint=None, celltype_model_checkpoint=None, gene_exp_model_checkpoint=None):

        traincfg = TrainConfig()

        self.trainer = DDPMTrainer(None, None, traincfg)
        self.location_model_checkpoint=location_model_checkpoint
        if location_model_checkpoint is not None:
            ckpt = torch.load(
                location_model_checkpoint,
                map_location=self.trainer.device,
                weights_only=False
            )
            self.trainer.model.load_state_dict(ckpt["model"])
            self.trainer.ema.shadow = ckpt["ema"]
        

        self.celltype_model = SkeletonCelltypeModel2(5274)#, num_features=3)

        if celltype_model_checkpoint is not None:
            self.celltype_model.load_model(celltype_model_checkpoint)

        
    def fit(self, training_adata_dir, val_adata_dir, cfg):

        # load using scanpy into a list, all of the h5ad files in the directory
        import os
        import scanpy as sc
        training_adata = []
        val_adata = []
        for file in os.listdir(training_adata_dir):
            if file.endswith((".h5ad", ".slaf")):
                adata = _read_slice(os.path.join(training_adata_dir, file))
                training_adata.append(adata)
        for file in os.listdir(val_adata_dir):
            if file.endswith((".h5ad", ".slaf")):
                adata = _read_slice(os.path.join(val_adata_dir, file))
                val_adata.append(adata)

        training_adata = sc.concat(training_adata)
        val_adata = sc.concat(val_adata)

        self.celltype_model.fit(
            training_adata,
            val_adata=val_adata,
            epochs=1000,
            batch_size=1024,
        )


