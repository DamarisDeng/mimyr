from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import anndata as ad

from metrics import soft_accuracy  # Ensure this function is available


class FixedFourierFeatureEncoding(nn.Module):
    def __init__(self, input_dim, num_frequencies=10, include_input=False):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Create a list of frequency bands: [1, 2, 4, ..., 2^{num_frequencies-1}]
        self.register_buffer(
            "freq_bands", 2 ** torch.arange(num_frequencies).float()
        )  # [num_frequencies]

    def forward(self, x):
        # x: [batch, input_dim]
        x = x.unsqueeze(-1)  # [batch, input_dim, 1]
        x_proj = x * self.freq_bands  # [batch, input_dim, num_frequencies]
        x_proj = 2 * np.pi * x_proj

        sin = torch.sin(x_proj)
        cos = torch.cos(x_proj)

        encoded = torch.cat([sin, cos], dim=-1)  # [batch, input_dim, 2*num_frequencies]
        encoded = encoded.view(
            x.shape[0], -1
        )  # [batch, input_dim * 2 * num_frequencies]

        if self.include_input:
            return torch.cat(
                [x.squeeze(-1), encoded], dim=-1
            )  # [batch, input_dim + encoded_dim]
        else:
            return encoded


class RandomFourierFeatureEncoding(nn.Module):
    def __init__(self, input_dim, num_features=256, scale=10.0, include_input=False):
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.include_input = include_input

        # Random projection matrix B ~ N(0, scale^2)
        self.register_buffer("B", torch.randn(num_features, input_dim) * scale)

    def forward(self, x):
        # x: [batch, input_dim]
        x_proj = 2 * np.pi * x @ self.B.T  # [batch, num_features]
        features = torch.cat(
            [torch.sin(x_proj), torch.cos(x_proj)], dim=-1
        )  # [batch, 2*num_features]

        if self.include_input:
            return torch.cat(
                [x, features], dim=-1
            )  # [batch, input_dim + 2*num_features]
        else:
            return features


class Model(nn.Module):
    def __init__(
        self,
        num_features=3,
        num_rff_features=256,
        num_classes=1,
        rff_off=True,
        distance_transform_off=True,
        memory_off=True,
        attention_off=True,
        num_points=4,  # <-- NEW ARGUMENT
    ):
        super().__init__()
        self.rff_off = rff_off
        self.distance_transform_off = distance_transform_off
        self.memory_off = memory_off
        self.attention_off = attention_off
        self.attn_dim = 1024  # Dim for Q, K, V

        if not distance_transform_off:
            # self.origins = nn.Parameter(torch.rand(num_points, 3) * 10)  # <-- NEW ORIGIN POINTS
            self.origins = nn.Parameter(
                torch.tensor([[5.0, 2, 5], [5, 4, 5], [5, 6, 5], [5, 5, 4]])
            )

        if not memory_off:
            self.memory = nn.Parameter(torch.rand(128).float())
            num_features += 128

        if not attention_off:
            self.attention_memory = nn.Parameter(torch.randn(128, num_features))
            self.to_q = nn.Linear(num_features, self.attn_dim)
            self.to_k = nn.Linear(num_features, self.attn_dim)
            self.to_v = nn.Linear(num_features, self.attn_dim)
            num_features = self.attn_dim

        if rff_off:
            self.network = nn.Sequential(
                nn.Linear(
                    4, 1024
                ),  # num_features if distance_transform_off else num_points+1, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_classes),
            )
        else:
            self.fourier = FixedFourierFeatureEncoding(
                input_dim=3, num_frequencies=100, include_input=True
            )
            ff_dim = 3 * 2 * 100 + num_features
            self.network = nn.Sequential(
                nn.Linear(ff_dim, 1024),
                nn.LeakyReLU(),
                *[
                    nn.Sequential(nn.Linear(1024, 1024), nn.LeakyReLU())
                    for _ in range(1)
                ],
                nn.Linear(1024, num_classes),
            )

    def forward(self, xyz):
        if not self.distance_transform_off:
            # Compute distances to each origin point
            # xyz: [B, 3], origins: [num_points, 3] → distances: [B, num_points]
            diffs = xyz[:, None, :3] - self.origins[None, :, :]  # [B, P, 3]
            distances = torch.norm(diffs, dim=-1)  # [B, P]
        else:
            distances = xyz
        if xyz.shape[1] > 3:
            distances = torch.cat([distances, xyz[:, 3:]], dim=-1)

        # distances=torch.cat([distances,xyz[3:]],dim=-1)

        if not self.memory_off:
            distances = torch.cat(
                [self.memory.expand(xyz.shape[0], -1), distances], dim=-1
            )

        if not self.attention_off:
            Q = self.to_q(distances).unsqueeze(1)  # [B, 1, attn_dim]
            K = self.to_k(self.attention_memory).unsqueeze(0)  # [1, T, attn_dim]
            V = self.to_v(self.attention_memory).unsqueeze(0)  # [1, T, attn_dim]

            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (
                self.attn_dim**0.5
            )  # [B, 1, T]
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, 1, T]
            attended = torch.matmul(attn_weights, V)  # [B, 1, attn_dim]
            distances = attended.squeeze(1)  # [B, attn_dim]

        if self.rff_off:
            return self.network(distances).squeeze(-1)
        else:
            encoded = self.fourier(distances)
            return self.network(encoded).squeeze(-1)


def margin_clipped_cross_entropy(outputs, targets, threshold=0.4):
    """
    outputs: [B, C] raw logits
    targets: [B] long
    """
    log_probs = torch.log_softmax(outputs, dim=1)  # [B, C]
    true_log_probs = log_probs[torch.arange(outputs.size(0)), targets]  # [B]

    # Cross-entropy is -log(p). So we mask out values below threshold
    losses = -true_log_probs
    mask = (losses > threshold).float()
    return (losses * mask).sum() / mask.sum().clamp(min=1.0)  # avoid divide-by-zero


def zero_grad_on_correct(outputs, targets):
    """
    outputs: [B, C] raw logits
    targets: [B] long
    """
    log_probs = torch.log_softmax(outputs, dim=1)  # [B, C]
    preds = torch.argmax(log_probs, dim=1)  # [B]
    correct_mask = (
        preds != targets
    ).float()  # [B] — 1 for incorrect predictions, 0 for correct

    true_log_probs = log_probs[torch.arange(outputs.size(0)), targets]  # [B]
    losses = -true_log_probs * correct_mask  # zero out correct predictions
    return losses.sum() / correct_mask.sum().clamp(min=1.0)  # avoid divide by 0


class CelltypeModel(nn.Module):
    def __init__(
        self,
        slices,
        n_classes,
        val_slice=None,
        learning_rate=0.001,
        batch_size=32,
        epochs=10,
        device=None,
    ):
        super(CelltypeModel, self).__init__()

        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.all_slices = ad.concat(slices)
        self.val_slice = val_slice
        self.best_model = None

        # Labels and inputs
        # parcellation_labels = self.all_slices.obs["parcellation_structure"].astype("category").cat.codes.values
        # parcellation_tensor = torch.tensor(parcellation_labels, dtype=torch.long).to(self.device)
        # one_hot_parcellation = torch.nn.functional.one_hot(parcellation_tensor).float()
        # self.parcellation_categories = self.all_slices.obs["parcellation_structure"].astype("category").cat.categories

        spatial_tensor = torch.tensor(
            self.all_slices.obsm["aligned_spatial"], dtype=torch.float32
        ).to(self.device)
        # density_tensor = torch.tensor(self.all_slices.obs["density"], dtype=torch.float32).to(self.device)
        # entropy_tensor = torch.tensor(self.all_slices.obs["entropy"], dtype=torch.float32).to(self.device)
        # pca_tensor = torch.tensor(self.all_slices.obsm["pca"], dtype=torch.float32).to(self.device)

        self.x = spatial_tensor  # torch.cat([spatial_tensor,entropy_tensor.unsqueeze(-1)],dim=-1)
        self.y = torch.tensor(self.all_slices.obs["token"].values, dtype=torch.long).to(
            self.device
        )

        self.model = Model(
            num_features=self.x.shape[-1],
            num_classes=n_classes,
            distance_transform_off=False,
            memory_off=True,
            rff_off=True,
            attention_off=True,
        ).to(self.device)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        # self.loss_fn = nn.CrossEntropyLoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Compute class-balanced weights
        alpha = 0.5  # soft reweighting factor; try 0.3–0.7 for a good tradeoff

        class_counts = torch.bincount(self.y)
        class_freqs = class_counts.float() / class_counts.sum()

        # Soft inverse frequency weighting
        class_weights = 1.0 / (class_freqs**alpha + 1e-6)
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()  # weight=class_weights)

        # self.loss_fn = margin_clipped_cross_entropy
        # self.loss_fn = zero_grad_on_correct

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(self):
        dataset = TensorDataset(self.x, self.y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        best_val_score = -1
        val_scores = []
        early_stop_patience = 5  # stop if no improvement in last 5 val checks
        epochs_since_improvement = 0

        for epoch in range(self.epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            if self.val_slice and (epoch + 1) % 5 == 0:
                val_score = self.evaluate_val()
                val_scores.append(val_score)
                print(
                    f"🔍 Validation soft accuracy at epoch {epoch+1}: {val_score:.4f}"
                )

                if val_score > best_val_score:
                    best_val_score = val_score
                    self.best_model = deepcopy(self.model)
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

                if epochs_since_improvement >= early_stop_patience:
                    print("⏹️ Early stopping triggered due to no improvement.")
                    break

            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.loss_fn(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total
            print(
                f"🚀 Epoch [{epoch+1}/{self.epochs}] - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}"
            )

        # At the end: restore best model and save weights
        if hasattr(self, "best_model"):
            self.model = self.best_model
            torch.save(self.model.state_dict(), "best_model.pt")
            print("💾 Best model restored and saved to best_model.pt")

    def evaluate_val(self):
        with torch.no_grad():
            val_x_full = torch.tensor(
                self.val_slice.obsm["aligned_spatial"], dtype=torch.float32
            ).to(self.device)
            n = val_x_full.shape[0]
            sample_size = max(1, n // 100)  # at least 1
            idx = np.random.choice(n, sample_size, replace=False)

            val_x = val_x_full[idx]
            # density_tensor = torch.tensor(self.val_slice.obs["entropy"].to_numpy(), dtype=torch.float32).to(self.device)[idx]
            # val_x = torch.cat([val_x, density_tensor.unsqueeze(-1)], dim=-1)

            outputs = self.model(val_x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = [
                np.random.choice(probs.shape[1], p=probs[i])
                for i in range(probs.shape[0])
            ]

        gt_celltypes = self.val_slice.obs["token"].to_numpy()[idx].tolist()
        gt_positions = self.val_slice.obsm["aligned_spatial"][idx]
        pred_positions = self.val_slice.obsm["aligned_spatial"][idx]
        pred_celltypes = preds

        return soft_accuracy(
            gt_celltypes, gt_positions, pred_celltypes, pred_positions, k=20
        )

    def get_token_distr(self, index):
        self.model.eval()
        with torch.no_grad():
            input_tensor = self.x[index].unsqueeze(0).to(self.device)
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
        return probabilities.cpu().numpy()

    def sample_output(self, x):
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device).unsqueeze(0)
            output = self.model(x_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten()
            sampled_token = np.random.choice(len(probabilities), p=probabilities)
        return sampled_token

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"📦 Loaded model weights from {path}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"💾 Saved model weights to {path}")


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from metrics import soft_accuracy  # optional, only if you keep it


class SkeletonCelltypeModel(nn.Module):
    def __init__(self, n_classes, num_features=3, learning_rate=0.001, device=None):
        super(SkeletonCelltypeModel, self).__init__()

        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Core classifier
        self.model = Model(
            num_features=num_features,
            num_classes=n_classes,
            distance_transform_off=False,
            memory_off=True,
            rff_off=True,
            attention_off=True,
        ).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.model(x)

    def get_token_distr(self, x):
        """Return probability distribution over classes for a single input tensor [F]."""
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device).unsqueeze(0)  # [1, F]
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def sample_output(self, x):
        """Sample one class index from the predicted distribution."""
        probs = self.get_token_distr(x).flatten()
        return np.random.choice(len(probs), p=probs)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
        self.model.to(self.device)
        print(f"📦 Loaded model weights from {path}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"💾 Saved model weights to {path}")


from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
import torch

from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from metrics import soft_accuracy  # keep this import


class SkeletonCelltypeModel2(nn.Module):
    def __init__(self, n_classes, num_features=3, learning_rate=0.001, device=None):
        super(SkeletonCelltypeModel2, self).__init__()

        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = Model(
            num_features=num_features,
            num_classes=n_classes,
            distance_transform_off=False,
            memory_off=True,
            rff_off=True,
            attention_off=True,
        ).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.best_model = None

    def forward(self, x):
        return self.model(x)

    def fit(self, train_adata, val_adata=None, batch_size=32, epochs=10, early_stop_patience=5,
            neighbor_tokens=None, neighborhood_weight=0.0):
        """Train the position -> cell-type classifier.

        neighbor_tokens / neighborhood_weight enable the neighborhood-KL objective:
        instead of (only) the one-hot label, fit each cell's softmax to the cell-type
        DISTRIBUTION of its spatial neighborhood. This is soft cross-entropy =
        forward KL(neighborhood || model), which is mode-COVERING (won't drop the
        tail), and teaches the position->local-composition map so rare types land
        where GT concentrates them rather than at a uniform base rate.

        neighbor_tokens: (N, k) int array of each train cell's k neighbor token-ids
                         (same row order as train_adata), precomputed by the caller.
        neighborhood_weight (lambda in [0,1]): loss = lambda * neighborhood_CE
                         + (1 - lambda) * hard_CE. 0 = original hard-label training
                         (byte-identical); 1 = pure neighborhood distribution.
        """
        X_train = torch.tensor(
            train_adata.obsm["aligned_spatial"], dtype=torch.float32
        ).to(self.device)
        y_train = torch.tensor(train_adata.obs["token"].values, dtype=torch.long).to(
            self.device
        )

        use_nbhd = neighbor_tokens is not None and neighborhood_weight > 0.0
        if use_nbhd:
            assert neighbor_tokens.shape[0] == X_train.shape[0], (
                f"neighbor_tokens rows {neighbor_tokens.shape[0]} != train cells {X_train.shape[0]}"
            )
            # int32 on-device to halve memory vs int64; cast per-batch for gather.
            nbr_train = torch.as_tensor(neighbor_tokens, dtype=torch.int32).to(self.device)
            dataset = TensorDataset(X_train, y_train, nbr_train)
            print(f"🧭 neighborhood-KL objective ON: k={neighbor_tokens.shape[1]} "
                  f"lambda={neighborhood_weight}")
        else:
            dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_score = -1
        val_scores = []
        epochs_since_improvement = 0

        for epoch in range(epochs):
            total_loss, correct, total = 0.0, 0, 0

            if val_adata is not None and (epoch + 1) % 1 == 0:
                val_score = self.evaluate_val(val_adata)
                val_scores.append(val_score)
                print(
                    f"🔍 Validation soft accuracy at epoch {epoch+1}: {val_score:.4f}"
                )

                if val_score > best_val_score:
                    best_val_score = val_score
                    self.best_model = deepcopy(self.model)
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

                if epochs_since_improvement >= early_stop_patience:
                    print("⏹️ Early stopping triggered due to no improvement.")
                    break

            for batch in dataloader:
                if use_nbhd:
                    xb, yb, nbrb = batch
                else:
                    xb, yb = batch
                self.optimizer.zero_grad()
                outputs = self.model(xb)
                if use_nbhd:
                    # forward-KL / soft-CE to the neighborhood distribution, computed
                    # exactly as the mean per-neighbor CE (avoids materializing dense
                    # C-wide targets): -(1/k) sum_n log softmax(logits)[token_n].
                    logp = torch.log_softmax(outputs, dim=1)
                    nbhd_ce = -logp.gather(1, nbrb.long()).mean()
                    if neighborhood_weight >= 1.0:
                        loss = nbhd_ce
                    else:
                        hard_ce = self.loss_fn(outputs, yb)
                        loss = neighborhood_weight * nbhd_ce + (1.0 - neighborhood_weight) * hard_ce
                else:
                    loss = self.loss_fn(outputs, yb)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

            avg_loss = total_loss / len(dataloader)
            acc = correct / total
            print(
                f"🚀 Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Accuracy: {acc:.4f}"
            )

        if self.best_model is not None:
            self.model = self.best_model
            torch.save(self.model.state_dict(), "best_model.pt")
            print("💾 Best model restored and saved to best_model.pt")

    def evaluate_val(self, val_adata, sample_size=50000):
        # Low-variance validation score for model selection. The original version
        # drew a fresh random n//100 subset each call AND sampled predictions
        # stochastically, so the early-stopping signal was dominated by noise and
        # tripped prematurely. Here we cache a fixed subset across epochs and use
        # deterministic argmax predictions, so the score reflects real learning.
        n = val_adata.n_obs
        if getattr(self, "_val_eval_idx", None) is None or len(self._val_eval_idx) > n:
            rng = np.random.default_rng(0)
            k = min(sample_size, n)
            self._val_eval_idx = np.sort(rng.choice(n, size=k, replace=False))
        idx = self._val_eval_idx

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            val_x = torch.tensor(
                val_adata.obsm["aligned_spatial"][idx], dtype=torch.float32
            ).to(self.device)
            outputs = self.model(val_x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
        if was_training:
            self.model.train()

        gt_celltypes = val_adata.obs["token"].to_numpy()[idx].tolist()
        positions = val_adata.obsm["aligned_spatial"][idx]

        return soft_accuracy(gt_celltypes, positions, preds, positions, k=20)

    def get_token_distr(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device).unsqueeze(0)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def sample_output(self, x):
        probs = self.get_token_distr(x).flatten()
        return np.random.choice(len(probs), p=probs)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"📦 Loaded model weights from {path}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"💾 Saved model weights to {path}")
