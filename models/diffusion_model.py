import sys

sys.path.append("../..")
sys.path.append("../generative_transformer")
from data_loader import SliceDataLoader

import itertools
import math
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import anndata as ad


# ---------------- Polynomial features ----------------
def _all_exponent_tuples(input_dim: int, degree: int):
    exps = []
    for total in range(degree + 1):
        for bars in itertools.combinations_with_replacement(range(input_dim), total):
            e = [0] * input_dim
            for b in bars:
                e[b] += 1
            exps.append(tuple(e))
    return exps


@dataclass
class TrainConfig:
    degree: int = 3
    hidden_sizes: tuple = ((512, 512),)  # (1024, 2048, 4096, 2048, 1024)
    activation: str = "relu"
    batchnorm: bool = False
    dropout: float = 0.0
    feature_type: str = "none"
    num_rff_features: int = 256
    rff_gamma: float = 100.0
    rff_seed: int | None = None
    n_timesteps: int = 1000
    schedule_type: str = "cosine"
    beta_start: float = 1e-10
    beta_end: float = 1e-9
    cosine_s: float = 0.008
    batch_size: int = 4096
    lr: float = 2e-4
    weight_decay: float = 0
    epochs: int = 200
    grad_clip: float = None
    ema_decay: float = 0.999
    # checkpoint basename prefix (epoch is appended). Override to avoid clobbering
    # other DDPM checkpoints, e.g. a Zhuang-1 location retrain.
    ckpt_prefix: str = "smoothtune2_conditional_ddpm_2d_checkpoint"
    # whether train() renders the in-loop 200k-point sample figures. Off = much
    # faster headless training (the figures were never saved to disk anyway).
    viz: bool = True


# ---------------- Timestep embedding ----------------
class TimestepEmbedding(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.register_buffer(
            "freqs", torch.exp(torch.linspace(0, math.log(10000.0), steps=dim // 2))
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.float().unsqueeze(1) * self.freqs.view(1, -1)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb


# ---------------- Featureizers ----------------
class PolynomialFeatures(nn.Module):
    def __init__(self, input_dim: int, degree: int):
        super().__init__()
        exps = _all_exponent_tuples(input_dim, degree)
        self.register_buffer("exponents", torch.tensor(exps, dtype=torch.long))

    @property
    def n_output_features_(self) -> int:
        return int(self.exponents.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(1)
        e = self.exponents.unsqueeze(0).to(x.device)
        monomials = torch.pow(x_expanded, e)
        return monomials.prod(dim=-1)


class IdentityFeatures(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    @property
    def n_output_features_(self) -> int:
        return self.input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class RandomFourierFeatures(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_features: int = 256,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__()
        self._num_features = int(num_features)
        self.gamma = float(gamma)

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(int(seed))

        W = torch.randn(self._num_features, input_dim, generator=g) * math.sqrt(
            2.0 * self.gamma
        )
        b = torch.rand(self._num_features, generator=g) * (2.0 * math.pi)

        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.scale = math.sqrt(2.0 / self._num_features)

    @property
    def n_output_features_(self) -> int:
        return self._num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.W.t() + self.b
        return self.scale * torch.cos(proj)


# ---------------- Model ----------------
class NoisePredictor(nn.Module):
    def __init__(
        self,
        input_dim=2,  # Now 2D
        degree=3,
        hidden_sizes=(512, 512),
        activation="silu",
        t_emb_dim=32,
        batchnorm=False,
        dropout=0.0,
        out_dim=2,  # Now 2D
        feature_type="poly",
        num_rff_features=256,
        rff_gamma=1.0,
        rff_seed=None,
        cond_dim=0,
    ):
        super().__init__()

        if feature_type == "poly":
            self.featurizer = PolynomialFeatures(input_dim, degree)
        elif feature_type == "rff":
            self.featurizer = RandomFourierFeatures(
                input_dim=input_dim,
                num_features=num_rff_features,
                gamma=rff_gamma,
                seed=rff_seed,
            )
        elif feature_type == "none":
            self.featurizer = IdentityFeatures(input_dim)
        else:
            raise ValueError("Only poly/rff/none implemented here")

        feat_dim = self.featurizer.n_output_features_
        self.t_emb = TimestepEmbedding(dim=t_emb_dim)

        acts = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "leakyrelu": lambda: nn.LeakyReLU(0.1),
            "tanh": nn.Tanh,
        }
        Act = acts.get(activation.lower(), nn.SiLU)

        in_dim = feat_dim + cond_dim + t_emb_dim
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h)]
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

        self.cond_dim = cond_dim

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        feats_x = self.featurizer(x)
        t_emb = self.t_emb(t)
        feats = torch.cat([feats_x, cond, t_emb], dim=-1)
        return self.net(feats)


# ---------------- EMA ----------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: p.detach().clone() for k, p in model.named_parameters()}
        for v in self.shadow.values():
            v.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, p in model.named_parameters():
            self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for k, p in model.named_parameters():
            p.copy_(self.shadow[k])


# ---------------- Trainer ----------------
class DDPMTrainer:
    def __init__(self, coords_np: np.ndarray, cond_np: np.ndarray, cfg: TrainConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if coords_np is None:
            self.mean = np.array([[5.0749707, 5.124198]])
            self.std = np.array([[2.2624643, 2.9367082]])
            print("Skipping data loading as coords_np is None")
        else:
            # normalize 2D coords
            # Store original 3D coords for reconstruction
            self.coords_3d_np = coords_np.copy()

            # Extract only x,y for training (drop z)
            coords_2d = coords_np[:, :2]  # (N, 2)

            self.mean = coords_2d.mean(axis=0, keepdims=True)
            self.std = coords_2d.std(axis=0, keepdims=True) + 1e-8

            self.mean = np.array([[5.0749707, 5.124198]])
            self.std = np.array([[2.2624643, 2.9367082]])

            x = (coords_2d - self.mean) / self.std

        # normalize conds
        if cond_np is None:
            self.cond_mean = np.array(
                [[5.138717, 3.9753077, 7.1279926, 0.33107758, 0.0, 0.6689224]]
            )
            self.cond_std = np.array(
                [
                    [
                        2.3946190e00,
                        2.0084219e00,
                        3.1232784e00,
                        4.7130716e-01,
                        9.9999999e-09,
                        4.7130716e-01,
                    ]
                ]
            )
            print("Skipping cond loading as cond_np is None")

        else:
            self.cond_mean = cond_np.mean(axis=0, keepdims=True)
            self.cond_std = cond_np.std(axis=0, keepdims=True) + 1e-8
            self.cond_mean = np.array(
                [[5.138717, 3.9753077, 7.1279926, 0.33107758, 0.0, 0.6689224]]
            )
            self.cond_std = np.array(
                [
                    [
                        2.3946190e00,
                        2.0084219e00,
                        3.1232784e00,
                        4.7130716e-01,
                        9.9999999e-09,
                        4.7130716e-01,
                    ]
                ]
            )
            print("Skipping cond loading as cond_np is None")

            c = (cond_np - self.cond_mean) / self.cond_std

        if coords_np is not None:
            self.data = torch.tensor(x, dtype=torch.float32)
            self.cond = torch.tensor(c, dtype=torch.float32)

            dataset = TensorDataset(self.data, self.cond)
            self.loader = DataLoader(
                dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False
            )

        self.cond_np = cond_np

        # schedule builder
        def _build_schedules(cfg: TrainConfig):
            T = cfg.n_timesteps
            schedule_type = getattr(cfg, "schedule_type", "linear").lower()

            if schedule_type == "linear":
                betas = torch.linspace(
                    cfg.beta_start, cfg.beta_end, T, dtype=torch.float64
                ).clamp(1e-8, 0.999)
                alphas = 1.0 - betas
                alpha_bars = torch.cumprod(alphas, dim=0)

            elif schedule_type == "cosine":
                s = float(getattr(cfg, "cosine_s", 0.008))
                steps = torch.arange(T + 1, dtype=torch.float64)

                def f(u):
                    return torch.cos((u * math.pi / 2.0)).pow(2)

                u = ((steps / T) ** 3 + s) / (1.0 + s)
                abar = f(u)
                abar = abar / abar[0].clamp_min(1e-20)

                betas = (1.0 - (abar[1:] / abar[:-1]).clamp_min(1e-20)).clamp(
                    1e-8, 0.999
                )
                alphas = 1.0 - betas
                alpha_bars = torch.cumprod(alphas, dim=0)

            else:
                raise ValueError("schedule_type must be 'linear' or 'cosine'")

            return (
                betas.to(torch.float32),
                alphas.to(torch.float32),
                alpha_bars.to(torch.float32),
            )

        betas, alphas, alpha_bars = _build_schedules(cfg)
        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alpha_bars = alpha_bars.to(self.device)

        self.model = NoisePredictor(
            input_dim=2,  # 2D now
            degree=cfg.degree,
            hidden_sizes=cfg.hidden_sizes,
            activation=cfg.activation,
            batchnorm=cfg.batchnorm,
            dropout=cfg.dropout,
            out_dim=2,  # 2D output
            feature_type=cfg.feature_type,
            cond_dim=6,
        ).to(self.device)

        self.ema = EMA(self.model, decay=cfg.ema_decay)
        self.opt = optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    def train(self):
        self.model.train()
        for epoch in range(1, self.cfg.epochs + 1):
            total_loss, n_batches = 0, 0

            for x_batch, cond_batch in self.loader:
                x0 = x_batch.to(self.device)
                cond = cond_batch.to(self.device)
                B = x0.shape[0]

                t = torch.randint(0, self.cfg.n_timesteps, (B,), device=self.device)
                noise = torch.randn_like(x0)

                alpha_bar_t = self.alpha_bars[t].view(-1, 1)
                x_noisy = (
                    torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
                )

                pred = self.model(x_noisy, t, cond)
                loss = ((pred - noise) ** 2).mean()

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.grad_clip:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                self.opt.step()
                self.ema.update(self.model)

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)
            print(f"Epoch {epoch:03d} | loss {avg_loss:.6f}")

            if epoch % 5 == 0:
                ckpt_name = f"model_checkpoints/{self.cfg.ckpt_prefix}_{epoch}.pt"
                if epoch % 20 == 0:
                    torch.save(
                        {
                            "model": self.model.state_dict(),
                            "ema": self.ema.shadow,
                            "opt": self.opt.state_dict(),
                            "cfg": self.cfg,
                            "coord_mean": self.mean,
                            "coord_std": self.std,
                            "cond_mean": self.cond_mean,
                            "cond_std": self.cond_std,
                        },
                        ckpt_name,
                    )

                # ---- generate and log a figure (headless runs skip this) ----
                if getattr(self.cfg, "viz", True):
                    # Use fixed plane at z=7: point=(0,0,7), normal=(0,0,1)
                    plane_z7 = np.array([3.0, 4.0, 6.7, 0.0, 0.0, 1.0], dtype=np.float32)
                    samples = self.sample(
                        200000,
                        use_ema=False,
                        cond_vec=plane_z7,
                    )
                    plt.figure()
                    plt.scatter(samples[:, 0], samples[:, 1], s=0.01, alpha=0.5)
                    plt.xlim(0, 12)
                    plt.ylim(0, 8)
                    plt.title(f"Generated 2D samples at epoch {epoch} (z=7 plane)")
                    plt.close()

                    plane_z7 = np.array([5.0, 4.0, 7.0, 1.0, 0.0, 0.0], dtype=np.float32)
                    samples = self.sample(
                        200000,
                        use_ema=False,
                        cond_vec=plane_z7,
                    )
                    plt.figure()
                    plt.scatter(samples[:, 0], samples[:, 1], s=0.01, alpha=0.5)
                    plt.xlim(0, 12)
                    plt.ylim(0, 8)
                    plt.title(f"Generated 2D samples at epoch {epoch} (z=7 plane)")
                    plt.close()

    @torch.no_grad()
    def sample(
        self, n_samples: int, use_ema=True, cond_vec=None, small_t_threshold=-1
    ) -> np.ndarray:
        """
        Generate 2D samples (x, y).
        To get 3D coordinates, extract z from the plane condition.
        """
        model = NoisePredictor(
            input_dim=2,
            degree=self.cfg.degree,
            hidden_sizes=self.cfg.hidden_sizes,
            activation=self.cfg.activation,
            batchnorm=self.cfg.batchnorm,
            dropout=self.cfg.dropout,
            out_dim=2,
            feature_type=self.cfg.feature_type,
            cond_dim=6,
        ).to(self.device)
        model.load_state_dict(self.model.state_dict())
        if use_ema:
            self.ema.copy_to(model)
        model.eval()

        # condition
        if cond_vec is None:
            cond = self.cond[0:1, :].repeat(n_samples, 1).to(self.device)
            cond_vec_unnorm = (
                self.cond[0].cpu().numpy() * self.cond_std + self.cond_mean
            )
        else:
            cond_vec = np.asarray(cond_vec, dtype=np.float32).reshape(1, -1)
            cond_vec_unnorm = cond_vec.copy()
            cond_norm = (cond_vec - self.cond_mean) / self.cond_std
            cond = torch.tensor(
                cond_norm, dtype=torch.float32, device=self.device
            ).repeat(n_samples, 1)

        x = torch.randn(n_samples, 2, device=self.device)  # 2D

        for t in reversed(range(self.cfg.n_timesteps)):
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            sqrt_recip_alpha = 1.0 / torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)

            z = torch.randn_like(x) if t > 0 else 0
            pred_noise = model(x, torch.full((n_samples,), t, device=self.device), cond)
            if t < small_t_threshold:
                z = 0

            x = (
                sqrt_recip_alpha
                * (x - (1 - alpha_t) / sqrt_one_minus_alpha_bar * pred_noise)
                + torch.sqrt(beta_t) * z
            )

        x_np = x.cpu().numpy() * self.std + self.mean  # (n_samples, 2)

        # Optionally add z coordinate from plane
        # Plane condition format: [px, py, pz, nx, ny, nz]
        z_coord = cond_vec_unnorm[0, 2]  # Extract z from plane point
        z_col = np.full((n_samples, 1), z_coord, dtype=np.float32)
        x_3d = np.concatenate([x_np, z_col], axis=1)  # (n_samples, 3)

        return x_3d

    @torch.no_grad()
    def sample_with_guidance(
        self,
        n_samples: int,
        potential_model: nn.Module,
        guidance_scale: float = 1.0,
        use_ema: bool = True,
        cond_vec=None,
        small_t_threshold=-1,
    ) -> np.ndarray:
        """
        Conditional classifier/potential-guided sampling.
        potential_model: torch.nn.Module, takes coords [B,3] -> (potential, grad)
        guidance_scale: multiplier on gradient of potential.
        cond_vec: conditioning vector (unnormalized)
        """
        # guidance_scale == 0 means "no guidance": delegate to the plain sampler
        # rather than multiplying the potential gradient by zero. Two reasons:
        # the unguided path stays bit-for-bit what it was before guidance was
        # re-wired (same RNG draw order -- sample() draws z before the model
        # forward, this function draws it after), and a non-finite KDE gradient
        # can't poison the result via 0 * inf = nan.
        if not guidance_scale or potential_model is None:
            return self.sample(
                n_samples,
                use_ema=use_ema,
                cond_vec=cond_vec,
                small_t_threshold=small_t_threshold,
            )

        # restore model with EMA
        model = NoisePredictor(
            input_dim=2,
            degree=self.cfg.degree,
            hidden_sizes=self.cfg.hidden_sizes,
            activation=self.cfg.activation,
            batchnorm=self.cfg.batchnorm,
            dropout=self.cfg.dropout,
            out_dim=2,
            feature_type=self.cfg.feature_type,
            cond_dim=6,
        ).to(self.device)
        model.load_state_dict(self.model.state_dict())
        if use_ema:
            self.ema.copy_to(model)
        model.eval()

        # condition
        if cond_vec is None:
            cond = self.cond[0:1, :].repeat(n_samples, 1).to(self.device)
            cond_vec_unnorm = (
                self.cond[0].cpu().numpy() * self.cond_std + self.cond_mean
            )
        else:
            cond_vec = np.asarray(cond_vec, dtype=np.float32).reshape(1, -1)
            cond_vec_unnorm = cond_vec.copy()
            cond_norm = (cond_vec - self.cond_mean) / self.cond_std
            cond = torch.tensor(
                cond_norm, dtype=torch.float32, device=self.device
            ).repeat(n_samples, 1)

        x = torch.randn(n_samples, 2, device=self.device)

        for t in reversed(range(self.cfg.n_timesteps)):
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            sqrt_recip_alpha = 1.0 / torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)

            # base prediction
            pred_noise = model(x, torch.full((n_samples,), t, device=self.device), cond)

            # denoised estimate
            x0_pred = (x - sqrt_one_minus_alpha_bar * pred_noise) / torch.sqrt(
                alpha_bar_t
            )

            # get ∇potential(x0_pred)
            x0_pred_world = x0_pred * torch.tensor(self.std).to(
                self.device
            ) + torch.tensor(self.mean).to(self.device)
            pot_val, grad = potential_model(x0_pred_world)
            grad = grad / torch.tensor(self.std).to(grad.device)
            guided_noise = pred_noise - guidance_scale * grad.float()

            z = torch.randn_like(x) if t > 0 else 0
            if t < small_t_threshold:
                z = 0

            x = (
                sqrt_recip_alpha
                * (x - (1 - alpha_t) / sqrt_one_minus_alpha_bar * guided_noise)
                + torch.sqrt(beta_t) * z
            )

        x_np = x.detach().cpu().numpy() * self.std + self.mean

        z_coord = cond_vec_unnorm[0, 2]  # z from plane
        z_col = np.full((n_samples, 1), z_coord, dtype=np.float32)
        x_3d = np.concatenate([x_np, z_col], axis=1)
        return x_3d
