from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import inspect
import math
from loguru import logger
import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from transformers.generation.utils import SampleDecoderOnlyOutput


class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.config = config

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        y = self.resid_dropout(self.c_proj(y))
        return y


def new_gelu(x):
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class MimyrConfig:
    block_size: int = 1000
    vocab_size: int = 1011
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    train_mode: str = "pretrain"
    expression_level: int = 10
    ele: int = 0
    bin_edges: np.ndarray = None
    hidden_regressor: bool = False  # if True, regressor takes transformer hidden state instead of bin logits
    continuous_coords: bool = False  # if True, <x>/<y>/<z> use a linear projection instead of the discrete wee lookup
    coord_token_ids: list = None  # token IDs for <x>, <y>, <z>; set from tokenizer at init time
    ordinal_bin_sigma: float = 0.0  # if >0, expression-bin loss uses a Gaussian soft target over bin indices (ordinal-aware) with this std; 0 = standard hard cross-entropy


class MimyrModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.expression_level is not None
        assert config.ele == 1
        self.config = config
        if config.bin_edges is not None:
            self.bin_edges = torch.tensor(config.bin_edges)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wee=nn.Embedding(
                    config.expression_level + 1, config.n_embd
                ),  # +1 for non gene tokens
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        if config.continuous_coords:
            self.coord_embed = nn.Linear(1, config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.epx_head = nn.Linear(
            config.n_embd, config.expression_level + 1, bias=False
        )  # +1 for 0 bin

        regressor_input_dim = config.n_embd if config.hidden_regressor else config.expression_level + 1
        self.epx_regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, 1),
        )

        if "LOCAL_RANK" not in os.environ or os.environ["LOCAL_RANK"] == "0":
            logger.info("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def to(self, device):
        model = super().to(device)
        if hasattr(model, "bin_edges"):
            model.bin_edges = model.bin_edges.to(device)
        return model

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def resize_token_embeddings(self, new_num_tokens: int):
        """
        Grow (or shrink) both the input token embeddings and the LM head
        to `new_num_tokens`, copying over old weights and randomly
        initializing any new rows.
        """
        old_emb = self.transformer.wte.weight.data
        old_num, emb_dim = old_emb.shape

        # 1) New embedding layer
        new_emb = nn.Embedding(new_num_tokens, emb_dim)
        with torch.no_grad():
            new_emb.weight[:old_num].copy_(old_emb)
        self.transformer.wte = new_emb

        # 2) New LM head (tied weights in many HF models)
        old_head = self.lm_head.weight.data  # shape (old_num, emb_dim)
        new_head = nn.Linear(emb_dim, new_num_tokens, bias=False)
        with torch.no_grad():
            new_head.weight[:old_num].copy_(old_head)
        self.lm_head = new_head

        # 3) Update config
        self.config.vocab_size = new_num_tokens

    def resize_expression_embeddings(self, new_expression_level: int):
        """
        Grow (or shrink) the expression‐level embedding `wee`
        to support values 0…new_expression_level (inclusive),
        copying old weights and randomly initializing any new rows.
        """
        old_emb = self.transformer.wee.weight.data
        old_num, emb_dim = old_emb.shape

        new_num = new_expression_level + 1

        # 1) new embedding
        new_emb = nn.Embedding(new_num, emb_dim)
        with torch.no_grad():
            new_emb.weight[:old_num].copy_(old_emb)
        self.transformer.wee = new_emb

        # 2) update config so that later calls (or re-initializations) know the new max
        self.config.expression_level = new_expression_level

        self.epx_head = nn.Linear(emb_dim, new_num, bias=False)

        regressor_input_dim = self.config.n_embd if self.config.hidden_regressor else new_num
        self.epx_regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, self.config.n_embd),
            nn.ReLU(),
            nn.Linear(self.config.n_embd, 1),
        )

    def forward(
        self,
        idx=None,
        inputs_embeds=None,
        targets=None,  # gene token IDs (B, T)
        xlen=None,
        x_prefix_len=None,
        x_expr=None,  # binned expression labels (B, T) — embedded as input
        y_expr=None,  # real-valued expression (B, T)
        x_expr_target=None,  # optional separate bin target for the bin-CE loss; if None, uses x_expr (teacher forcing). Lets the embedded input be self-generated (scheduled sampling) while the loss target stays GT.
        lambda_val: float = 1.0,
        return_hidden=False,
    ):

        if idx is not None:
            b, t = idx.size()
            tok_emb = self.transformer.wte(idx)
        if inputs_embeds is not None:
            tok_emb = inputs_embeds

        expr_emb = self.transformer.wee(x_expr)
        if self.config.continuous_coords and self.config.coord_token_ids:
            coord_ids = torch.tensor(
                self.config.coord_token_ids, dtype=torch.long, device=idx.device
            )
            coord_mask = (idx.unsqueeze(-1) == coord_ids).any(-1)  # (B, T)
            coord_vals = x_expr.float() / self.config.expression_level  # normalize bin to [0, 1]
            coord_embs = self.coord_embed(coord_vals.unsqueeze(-1))  # (B, T, n_embd)
            expr_emb = torch.where(coord_mask.unsqueeze(-1), coord_embs, expr_emb)
        x = self.transformer.drop(tok_emb + expr_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits_labels = self.lm_head(x)  # (B, T, vocab_size)
        logits_exp_bins = self.epx_head(x)  # (B, T, num_bins)
        regressor_input = x if self.config.hidden_regressor else logits_exp_bins
        logits_exp_real = self.epx_regressor(regressor_input)  # (B, T, 1)

        loss = loss_cls = loss_exp_bin = loss_exp_real = None
        eos_token_id = 0
        B, T, V = logits_labels.shape
        num_bins = logits_exp_bins.shape[-1]

        if targets is not None:
            ### --- Set Classification Loss ---

            # logits_labels: (B, T, V)
            # probs = F.softmax(logits_labels, dim=-1)       # (B, T, V)
            # pred_set_soft = probs.sum(dim=1)               # (B, V)

            # # True set vector (B, V)
            # mask = (targets != -100)
            # flat_indices = targets[mask]
            # batch_indices = torch.arange(B, device=targets.device).unsqueeze(1).expand(B, T)[mask]
            # true_set = torch.zeros((B, V), device=targets.device)
            # true_set.index_put_((batch_indices, flat_indices), torch.ones_like(flat_indices, dtype=torch.float), accumulate=True)
            # true_set = true_set.clamp(max=1.0)

            # loss_cls = F.binary_cross_entropy(pred_set_soft.clamp(min=1e-6, max=1.0), true_set)

            shift_logits = logits_labels[:, :-1, :].reshape(-1, V)  # (B*(T-1), V)
            shift_labels = targets[:, 1:].reshape(-1)  # (B*(T-1))
            loss_cls = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

            # targets_shifted = targets[:, 1:]                        # (B, T-1)
            # eos_mask = (targets_shifted == eos_token_id)            # (B, T-1)

            # first_eos_pos = eos_mask.float().cumsum(dim=1) == 1.0   # (B, T-1)
            # flat_logits = logits_labels[:, :-1, :].reshape(-1, V)   # (B*(T-1), V)
            # flat_first_eos = first_eos_pos.reshape(-1)              # (B*(T-1),)

            # eos_indices = flat_first_eos.nonzero(as_tuple=False).squeeze(-1)  # (N_valid,)
            # eos_logits = flat_logits[eos_indices]                              # (N_valid, V)

            # eos_targets = torch.full((eos_logits.size(0),), eos_token_id, dtype=torch.long, device=logits_labels.device)

            # if eos_logits.size(0) > 0:
            #     eos_loss = F.cross_entropy(eos_logits, eos_targets, reduction='mean')
            # else:
            #     eos_loss = torch.tensor(0.0, device=logits_labels.device)
            # loss_cls += eos_loss

            ### --- Set Binned Expression Loss ---
            # if x_expr is not None:

            #     P_gene = F.softmax(logits_labels, dim=-1)        # (B, T, V)
            #     P_bin  = F.softmax(logits_exp_bins, dim=-1)      # (B, T, num_bins)

            #     # Efficient computation: (B, V, T) x (B, T, num_bins) → (B, V, num_bins)
            #     pred_binned_expr = torch.bmm(P_gene.transpose(1, 2), P_bin)  # (B, V, num_bins)
            #     B, V, num_bins = pred_binned_expr.shape

            #     true_bin_labels = torch.full((B, V), -100, dtype=torch.long, device=targets.device)

            #     # Fill true labels using scatter
            #     mask = (targets != -100)
            #     flat_gene_ids = targets[mask]                    # (num_valid,)
            #     flat_bin_vals = x_expr[mask]                     # (num_valid,)
            #     flat_batch_ids = torch.arange(B, device=targets.device).unsqueeze(1).expand(B, T)[mask]

            #     # Assign to the true_bin_labels
            #     true_bin_labels.index_put_(
            #         (flat_batch_ids, flat_gene_ids),
            #         flat_bin_vals,
            #         accumulate=False
            #     )

            #     # Mask to select only genes that appeared in the true targets
            #     true_gene_mask = torch.zeros((B, V), dtype=torch.bool, device=targets.device)
            #     true_gene_mask.index_put_(
            #         (flat_batch_ids, flat_gene_ids),
            #         torch.ones_like(flat_gene_ids, dtype=torch.bool),
            #         accumulate=True
            #     )

            #     # Apply mask to loss
            #     # loss_exp_bin = F.cross_entropy(
            #     #     pred_binned_expr[true_gene_mask],       # (num_true_genes, num_bins)
            #     #     true_bin_labels[true_gene_mask]         # (num_true_genes,)
            #     # )
            #     loss_exp_bin = F.cross_entropy(
            #         pred_binned_expr.reshape(-1, num_bins),       # (V, num_bins)
            #         true_bin_labels.reshape(-1)        # (V,)
            #     )

            #     P_gene = F.softmax(logits_labels, dim=-1)                # (B, T, V)
            #     P_gene_T = P_gene.transpose(1, 2)                         # (B, V, T)

            #     # Step 2: Predicted real expression values
            #     pred_real = logits_exp_real.squeeze(-1)                  # (B, T)

            #     # Step 3: Weighted sum over time — use bmm
            #     # (B, V, T) x (B, T, 1) -> (B, V, 1) -> squeeze -> (B, V)
            #     pred_real_by_gene = torch.bmm(P_gene_T, pred_real.unsqueeze(-1)).squeeze(-1)  # (B, V)

            #     # Step 3: true real expression values
            #     true_real_by_gene = torch.zeros((B, V), device=targets.device)

            #     flat_real_vals = y_expr[mask]

            #     # Scatter the real values into the (B, V) matrix
            #     true_real_by_gene.index_put_(
            #         (flat_batch_ids, flat_gene_ids),
            #         flat_real_vals,
            #         accumulate=False
            #     )

            #     # Final masked MSE loss
            #     # loss_exp_real = F.mse_loss(
            #     #     pred_real_by_gene[true_gene_mask],
            #     #     true_real_by_gene[true_gene_mask]
            #     # )
            #     loss_exp_real = F.mse_loss(
            #         pred_real_by_gene,
            #         true_real_by_gene
            #     )

            if x_expr is not None:
                B, T, V = logits_exp_bins.shape
                # The bin-CE target is the GT bin sequence. Under scheduled sampling the
                # *embedded* x_expr may be (partly) self-generated, so the target is passed
                # separately; default to x_expr to preserve plain teacher-forcing behavior.
                bin_target = x_expr_target if x_expr_target is not None else x_expr
                x_expr_masked = bin_target.clone()
                x_expr_masked[targets == -100] = -100

                shift_exp_logits = logits_exp_bins[:, :-1, :].reshape(-1, V)
                shift_exp_labels = x_expr_masked[:, 1:].reshape(-1)

                sigma = getattr(self.config, "ordinal_bin_sigma", 0.0)
                if sigma and sigma > 0.0:
                    # Ordinal-aware bin loss: instead of a one-hot target (which treats
                    # bins as independent classes), use a Gaussian soft target centered
                    # on the true bin index. Predicting an adjacent bin is then penalized
                    # far less than a distant one. sigma -> 0 recovers hard cross-entropy.
                    # (Soft ordinal labels; cf. Diaz & Marathe, CVPR 2019.)
                    valid = shift_exp_labels != -100
                    if valid.any():
                        logits_v = shift_exp_logits[valid]            # (M, K)
                        labels_v = shift_exp_labels[valid].float()    # (M,)
                        bins = torch.arange(V, device=logits_v.device, dtype=torch.float)  # (K,)
                        soft_target = F.softmax(
                            -((bins.unsqueeze(0) - labels_v.unsqueeze(1)) ** 2) / (2.0 * sigma * sigma),
                            dim=-1,
                        )  # (M, K)
                        log_probs = F.log_softmax(logits_v, dim=-1)   # (M, K)
                        loss_exp_bin = -(soft_target * log_probs).sum(dim=-1).mean()
                    else:
                        loss_exp_bin = shift_exp_logits.sum() * 0.0   # no valid tokens; keep graph
                else:
                    loss_exp_bin = F.cross_entropy(
                        shift_exp_logits, shift_exp_labels, ignore_index=-100
                    )

            loss_exp_real = None
            if y_expr is not None:
                pred_vals = logits_exp_real.squeeze(-1)
                true_vals = y_expr.squeeze(-1) if y_expr.dim() == 3 else y_expr

                shift_pred = pred_vals[:, :-1].contiguous()
                shift_true = true_vals[:, 1:].contiguous()
                # Mask out prompt and padding positions (same mask as loss_exp_bin)
                real_mask = targets[:, 1:] != -100  # (B, T-1)
                if real_mask.any():
                    loss_exp_real = F.mse_loss(
                        shift_pred[real_mask], shift_true[real_mask], reduction="mean"
                    )
                else:
                    loss_exp_real = torch.tensor(0.0, device=shift_pred.device)

        # Final loss
        if (
            (loss_cls is not None)
            and (loss_exp_real is not None)
            and (loss_exp_bin is not None)
        ):
            loss = loss_cls + lambda_val * (loss_exp_real + loss_exp_bin)
        elif loss_cls is not None:
            loss = loss_cls
        elif loss_exp_real is not None and loss_exp_bin is not None:
            loss = lambda_val * (loss_exp_real + loss_exp_bin)

        if return_hidden:
            return logits_labels, logits_exp_bins, logits_exp_real, x
        else:
            return (
                logits_labels,
                logits_exp_bins,
                logits_exp_real,
                loss,
                loss_cls,
                loss_exp_bin,
                loss_exp_real,
            )

    # def forward(self,
    #             idx=None,
    #             inputs_embeds=None,
    #             targets=None,     # token IDs for attr‐prediction (shape [B])
    #             xlen=None,
    #             x_prefix_len=None,
    #             x_expr=None,      # input expression for embedding head
    #             y_expr=None,      # true expression values for regression loss (shape [B])
    #             lambda_val: float = 1.0,
    #             return_hidden=False,
    # ):

    #     if idx is not None:
    #         b, t = idx.size()
    #         tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

    #     if inputs_embeds is not None:
    #         tok_emb = inputs_embeds

    #     expr_emb = self.transformer.wee(x_expr) # expression embeddings of shape (b, t, n_embd)
    #     x = self.transformer.drop(tok_emb + expr_emb)

    #     for block in self.transformer.h:
    #         x = block(x)
    #     x = self.transformer.ln_f(x)

    #     logits_labels = self.lm_head(x)
    #     # logits_exp = self.epx_head(x).squeeze(-1)
    #     logits_exp_bins = self.epx_head(x)                    # (B, T, num_bins)
    #     logits_exp_real = self.epx_regressor(logits_exp_bins) # (B, T, 1)
    #     loss = None
    #     loss_cls = None
    #     loss_exp = None
    #     loss_exp_bin = None

    #     eos_token_id = 0
    #     eos_weight = 2.0

    #     # 2) Classification loss over all tokens (causal shift)
    #     if targets is not None:
    #         B, T, V = logits_labels.shape
    #         # shift so that at step t we predict targets[:,t] from x[:,t-1]
    #         shift_logits = logits_labels[:, :-1, :].reshape(-1, V)   # (B*(T-1), V)
    #         shift_labels = targets[:, 1:].reshape(-1)             # (B*(T-1))
    #         # loss_weights = torch.ones_like(shift_labels, dtype=torch.float)
    #         # loss_weights[shift_labels == eos_token_id] = eos_weight

    #         loss_cls = F.cross_entropy(
    #             shift_logits,
    #             shift_labels,
    #             ignore_index=-100,
    #             # weight=loss_weights.view(-1),
    #         )

    #     if y_expr is not None:
    #         B, T, V = logits_exp_bins.shape
    #         x_expr_masked = x_expr.clone()
    #         x_expr_masked[targets == -100] = -100

    #         shift_exp_logits = logits_exp_bins[:, :-1, :].reshape(-1, V)
    #         shift_exp_labels = x_expr_masked[:, 1:].reshape(-1)
    #         loss_exp_bin = F.cross_entropy(shift_exp_logits, shift_exp_labels, ignore_index=-100)

    #     loss_exp_real = None
    #     if y_expr is not None:
    #         pred_vals = logits_exp_real.squeeze(-1)
    #         true_vals = y_expr.squeeze(-1) if y_expr.dim() == 3 else y_expr

    #         shift_pred = pred_vals[:, :-1].contiguous()
    #         shift_true = true_vals[:, 1:].contiguous()
    #         loss_exp_real = F.mse_loss(shift_pred, shift_true, reduction='mean')

    #     # B, T, V = logits_exp.size()
    #     # flat_exp = logits_exp.view(-1, V)             # (B*T, V)
    #     # flat_reg = self.epx_regressor(flat_exp)       # (B*T, 1)
    #     # logits_reg = flat_reg.view(B, T, 1)           # (B, T, 1)

    #     # 3) Expression loss over all tokens (aligned or shifted as needed)
    #     # if y_expr is not None:
    #     #     # If you want causal‐style (predict y_expr[:,t] from x[:,t-1]):
    #     #     #   pred_vals = logits_exp[:, :-1].reshape(-1)
    #     #     #   true_vals = y_expr[:, 1:].reshape(-1)
    #     #     # Otherwise, to predict at each position:

    #     #     pred_vals = logits_reg.squeeze(-1)
    #     #     shift_pred = pred_vals[:,:-1].contiguous()
    #     #     # true_vals: ensure same shape
    #     #     true_vals = y_expr.squeeze(-1) if y_expr.dim()==3 else y_expr
    #     #     shift_true = true_vals[:,1:].contiguous()

    #     #     flat_pred = shift_pred.view(-1)
    #     #     flat_true = shift_true.view(-1)

    #     #     loss_exp = F.mse_loss(pred_vals, true_vals, reduction='mean')

    #     # if (loss_cls is not None) and (loss_exp is not None):
    #     #     loss = loss_cls + lambda_val * loss_exp
    #     # elif loss_cls is not None:
    #     #     loss = loss_cls
    #     # elif loss_exp is not None:
    #     #     loss = lambda_val * loss_exp

    #     if (loss_cls is not None) and (loss_exp_real is not None) and (loss_exp_bin is not None):
    #         loss = loss_cls + lambda_val * (loss_exp_real + loss_exp_bin)
    #     elif loss_cls is not None:
    #         loss = loss_cls
    #     elif loss_exp_real is not None and loss_exp_bin is not None:
    #         loss = lambda_val * (loss_exp_real + loss_exp_bin)

    #     if return_hidden:
    #         return logits_labels, logits_exp_bins, logits_exp_real, x
    #     else:
    #         return logits_labels, logits_exp_bins, logits_exp_real, loss, loss_cls, loss_exp_bin, loss_exp_real

    # @torch.no_grad()
    # def generate_cellGenesis(self,
    #                   input_ids,
    #                   expression_level,
    #                   max_new_tokens,
    #                   ignore_Idx = None,
    #                   top_k = None,
    #                   return_dict_in_generate = False,
    #                   return_hidden = False,
    #                   gamma = 1):

    #     scores = ()
    #     hidden_states = ()
    #     while True:
    #         idx_cond = input_ids
    #         if return_hidden:
    #             logits_cls,logits_exp,hidden = self(idx = idx_cond, x_expr = expression_level, return_hidden = True)
    #             hidden_states += (hidden,)
    #         else:
    #             logits_cls,logits_exp,_,_,_ = self(idx = idx_cond, x_expr = expression_level)

    #         logits_cls = logits_cls[:,-1,:] # (B,C)
    #         logits_exp = logits_exp[:,-1,:] # (B,C)

    #         if ignore_Idx is not None:
    #             # return logits, ignore_Idx
    #             logits_cls[:,ignore_Idx] = float('-inf')
    #         logits_cls[:,input_ids[0,:-1]] = float('-inf')
    #         if top_k is not None:
    #             v, _ = torch.topk(logits_cls, min(top_k, logits_cls.size(-1)))
    #             logits_cls[logits_cls < v[:, [-1]]] = float('-inf')

    #         next_token_scores = logits_cls

    #         if return_dict_in_generate:
    #             scores += (next_token_scores,)

    #         probs = F.softmax(logits_cls, dim=-1) #(B,C)
    #         probs[:,0] = gamma*probs[:,0]
    #         next_tokens = torch.multinomial(probs,num_samples=1) #(B,1)
    #         next_token_ele = logits_exp[torch.arange(logits_exp.size(0)),next_tokens.squeeze(1)].unsqueeze(1) # (B,1)
    #         bin_ele_next_token = torch.clamp(torch.round(next_token_ele), 0, 10).int()
    #         input_ids = torch.cat((input_ids,next_tokens),dim=1)
    #         expression_level = torch.cat((expression_level,bin_ele_next_token),dim=1)
    #         # check break condition
    #         if next_tokens == 0 or len(input_ids[0]) >= max_new_tokens:
    #             break

    #     if return_dict_in_generate:
    #         return SampleDecoderOutput(
    #             sequences=input_ids,
    #             scores=scores,
    #             hidden_states=hidden_states,
    #             expression=expression_level,
    #             )
    #     elif return_hidden:
    #         return input_ids,expression_level, hidden_states
    #     else:return input_ids, expression_level

    # @torch.no_grad()
    # def generate_cellGenesis(
    #     self,
    #     input_ids: torch.LongTensor,
    #     expression_level: torch.LongTensor,
    #     max_new_tokens: int,
    #     ignore_Idx=None,
    #     top_k=None,
    #     return_dict_in_generate=False,
    #     return_hidden=False,
    #     gamma: float = 1.0,
    # ):
    #     """
    #     Autoregressively generate gene tokens *and* real‐valued expression
    #     using your MLP regression head on top of the pretrained categorical head.
    #     """
    #     scores = ()           # if return_dict_in_generate
    #     hidden_states = ()    # if return_hidden
    #     real_expr = torch.tensor(expression_level.clone(), dtype=torch.float)

    #     # loop until EOS (token 0) or length
    #     while True:
    #         # 1) forward one step
    #         if return_hidden:
    #             logits_cls, logits_exp, hidden = self(
    #                 idx=input_ids,
    #                 x_expr=expression_level,
    #                 return_hidden=True
    #             )
    #             hidden_states += (hidden,)
    #         else:
    #             logits_cls, logits_exp, _, _, _ = self(
    #                 idx=input_ids,
    #                 x_expr=expression_level
    #             )

    #         # 2) grab the *last* time-step logits
    #         #    cls: (B, vocab_size), exp: (B,) real‐valued
    #         logits_cls = logits_cls[:, -1, :]           # (B, C)
    #         logits_exp = logits_exp[:, -1].float()      # (B, 1)

    #         # 3) mask out forbidden/seen tokens
    #         if ignore_Idx is not None:
    #             logits_cls[:, ignore_Idx] = float('-inf')
    #         seen = input_ids[0, :-1]
    #         logits_cls[:, seen] = float('-inf')

    #         # 4) top-k filtering
    #         if top_k is not None:
    #             v, _ = torch.topk(logits_cls, min(top_k, logits_cls.size(-1)), dim=-1)
    #             logits_cls[logits_cls < v[:, [-1]]] = float('-inf')

    #         if return_dict_in_generate:
    #             scores += (logits_cls,)

    #         # 5) sample next token
    #         probs       = F.softmax(logits_cls, dim=-1)
    #         probs[:, 0] = gamma * probs[:, 0]       # optional weighting for EOS
    #         next_token  = torch.multinomial(probs, num_samples=1)  # (B,1)

    #         next_expr_real = logits_exp  # (B,1)

    #         bin_next = torch.bucketize(next_expr_real, self.bin_edges)  # (B,)

    #         bin_next = torch.clamp(bin_next, 0, self.bin_edges.numel()-1)  # (B,1)

    #         # print(expression_level.shape, bin_next.shape, next_expr_real.shape)
    #         # 8) append to sequence
    #         input_ids        = torch.cat([input_ids, next_token], dim=1)
    #         expression_level = torch.cat([expression_level, bin_next], dim=1)
    #         real_expr = torch.cat([real_expr, next_expr_real], dim=1)
    #         # 9) stop if EOS or too long
    #         if (next_token == 0).all() or input_ids.size(1) >= max_new_tokens:
    #             break

    #     if return_dict_in_generate:
    #         return SampleDecoderOutput(
    #             sequences=input_ids,
    #             scores=scores,
    #             hidden_states=hidden_states,
    #             expression=expression_level,
    #         )
    #     elif return_hidden:
    #         return input_ids, expression_level, real_expr, hidden_states
    #     else:
    #         return input_ids, expression_level, real_expr

    @torch.no_grad()
    def generate_cellGenesis(
        self,
        input_ids: torch.LongTensor,
        expression_level: torch.LongTensor,
        max_new_tokens: int,
        ignore_Idx=None,
        top_k=None,
        return_dict_in_generate=False,
        return_hidden=False,
        gamma: float = 1.0,
        override_gene_sequence: Optional[torch.LongTensor] = None,
        override_expr_sequence: Optional[torch.LongTensor] = None,
        verbose: bool = False,
        select_max_index_margin: float = 0.0,
    ):
        """
        Autoregressively generate gene tokens *and* real-valued expression.
        Optionally use teacher-forced gene/expression tokens as model inputs,
        but always return the model's own predictions as outputs.
        """
        scores = ()
        hidden_states = ()

        # Start with empty predictions
        predicted_gene_tokens = []
        predicted_expression_bins = []
        predicted_real_expr = []

        B = input_ids.size(0)
        finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)

        while True:
            step_idx = input_ids.size(1)

            # 1) model forward
            if return_hidden:
                logits_cls, logits_exp_bins, logits_exp_real, hidden = self(
                    idx=input_ids, x_expr=expression_level, return_hidden=True
                )
                hidden_states += (hidden,)
            else:
                logits_cls, logits_exp_bins, logits_exp_real, _, _, _, _ = self(
                    idx=input_ids, x_expr=expression_level
                )
            if verbose == True:
                predicted_token_ids = torch.argmax(logits_cls, dim=-1)
                print(f"predicted token ids: {predicted_token_ids}")
            logits_cls = logits_cls[:, -1, :]  # (B, vocab)
            logits_exp_bins = logits_exp_bins[:, -1, :]  # (B, num_bins)
            logits_exp_real = logits_exp_real[:, -1].float()  # (B, 1)

            # 2) mask invalid tokens
            if ignore_Idx is not None:
                logits_cls[:, ignore_Idx] = float("-inf")
            seen = input_ids[0, :-1]
            logits_cls[:, seen] = float("-inf")

            # 3) top-k filter
            if top_k is not None:
                v, _ = torch.topk(logits_cls, min(top_k, logits_cls.size(-1)), dim=-1)
                logits_cls[logits_cls < v[:, [-1]]] = float("-inf")

            # mask tokens with index >= previous token (enforce strictly decreasing order)
            prev_tokens = input_ids[:, -1]  # (B,)
            token_range = torch.arange(logits_cls.size(-1), device=logits_cls.device).unsqueeze(0)
            logits_cls = logits_cls.masked_fill(token_range >= prev_tokens.unsqueeze(1), float("-inf"))

            if return_dict_in_generate:
                scores += (logits_cls,)

            # 4) sample model predictions
            probs = F.softmax(logits_cls, dim=-1)
            if select_max_index_margin > 0.0:
                # Among still-allowed (finite) tokens whose probability is within
                # `margin` of the top probability, pick the HIGHEST token index.
                # The argmax is always within the margin (diff 0), so this returns
                # the argmax unless a higher-index token is also within the margin —
                # it never falls back to sampling.
                forced_eos = probs[:, 0] >= 0.95
                max_prob = probs.max(dim=-1, keepdim=True).values  # (B,1)
                eligible = ((max_prob - probs) < select_max_index_margin) & torch.isfinite(logits_cls)  # (B,V)
                token_indices = torch.arange(
                    logits_cls.size(-1), device=logits_cls.device
                ).unsqueeze(0).expand_as(logits_cls)
                next_token_raw = token_indices.masked_fill(~eligible, -1).max(dim=-1, keepdim=True).values  # (B,1)
                next_token = torch.where(forced_eos.unsqueeze(1), next_token_raw.new_full((next_token_raw.size(0), 1), 0), next_token_raw)
            else:
                probs[:, 0] = gamma * probs[:, 0]
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Mark items that have generated EOS (0)
            newly_finished = next_token.squeeze(1) == 0
            finished |= newly_finished

            next_expr_real = logits_exp_real.contiguous()  # (B, 1)
            bin_next = torch.argmax(logits_exp_bins, dim=-1, keepdim=True)  # (B, 1)

            # 5) record model predictions
            predicted_gene_tokens.append(next_token)
            predicted_expression_bins.append(bin_next)
            predicted_real_expr.append(next_expr_real)

            # 6) prepare teacher-forced or predicted input for next step
            if (
                override_gene_sequence is not None
                and step_idx < override_gene_sequence.size(1)
            ):
                input_token_for_next_step = override_gene_sequence[
                    :, step_idx
                ].unsqueeze(1)
            else:
                input_token_for_next_step = next_token

            if (
                override_expr_sequence is not None
                and step_idx < override_expr_sequence.size(1)
            ):
                expr_bin_for_next_step = override_expr_sequence[:, step_idx].unsqueeze(
                    1
                )
            else:
                expr_bin_for_next_step = bin_next

            # 7) update inputs
            input_ids = torch.cat([input_ids, input_token_for_next_step], dim=1)
            expression_level = torch.cat(
                [expression_level, expr_bin_for_next_step], dim=1
            )

            if verbose == True:
                print(
                    f"next_pred_token: {next_token}, next_token: {input_token_for_next_step}, bin_next: {bin_next},"
                )
                print(f"input_ids: {input_ids}")
                print(logits_exp)
                print(f"expression level: {expression_level}")
                print("\n")

            # 8) stop condition
            if finished.all() or predicted_gene_tokens.__len__() >= max_new_tokens:
                break

        # concatenate predicted sequences
        predicted_gene_tokens = torch.cat(predicted_gene_tokens, dim=1)  # (B, T)
        predicted_expression_bins = torch.cat(
            predicted_expression_bins, dim=1
        )  # (B, T)
        predicted_real_expr = torch.cat(predicted_real_expr, dim=1)  # (B, T)

        if return_dict_in_generate:
            return SampleDecoderOutput(
                sequences=predicted_gene_tokens,
                scores=scores,
                hidden_states=hidden_states,
                expression=predicted_expression_bins,
                real_expression=predicted_real_expr,
            )
        elif return_hidden:
            return (
                predicted_gene_tokens,
                predicted_expression_bins,
                predicted_real_expr,
                hidden_states,
            )
        else:
            return predicted_gene_tokens, predicted_expression_bins, predicted_real_expr


@dataclass
class SampleDecoderOutput(SampleDecoderOnlyOutput):

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    expression: Optional[torch.LongTensor] = None
