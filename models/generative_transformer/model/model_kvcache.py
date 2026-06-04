from dataclasses import dataclass
from typing import Optional, Tuple, Union
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
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

    def forward(self, x, layer_past=None):
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        B, T, C = k.size()
        _, Tq, _ = q.size()

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        if Tq < k.size(2):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )

        y = y.transpose(1, 2).contiguous().view(B, Tq, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, (k, v)


def new_gelu(x):
    return (
        0.5
        * x
        * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))
    )


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(new_gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, layer_past=None):
        a, kv = self.attn(self.ln_1(x), layer_past)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, kv


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
    ele: int = 1
    bin_edges: torch.Tensor = None


class MimyrModel_kv(nn.Module):
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
        )  # expr level

        self.epx_regressor = nn.Sequential(
            nn.Linear(config.expression_level + 1, config.n_embd),  # map vocab→hidden
            nn.ReLU(),
            nn.Linear(config.n_embd, 1),  # hidden→scalar
        )

        if "LOCAL_RANK" not in os.environ or os.environ["LOCAL_RANK"] == "0":
            logger.info("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
        # super().__init__()
        # assert config.ele == 1
        # self.config = config
        # if config.bin_edges is not None:
        #     self.bin_edges = config.bin_edges

        # self.transformer = nn.ModuleDict(dict(
        #     wte=nn.Embedding(config.vocab_size, config.n_embd),
        #     wee=nn.Embedding(config.expression_level + 1, config.n_embd),
        #     drop=nn.Dropout(config.dropout),
        #     h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        #     ln_f=LayerNorm(config.n_embd, bias=config.bias)
        # ))

        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.epx_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.epx_regressor = nn.Sequential(
        #     nn.Linear(config.vocab_size, config.n_embd),
        #     nn.ReLU(),
        #     nn.Linear(config.n_embd, 1)
        # )

        # if 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] == '0':
        #     logger.info(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def to(self, device):
        model = super().to(device)
        if hasattr(model, "bin_edges"):
            model.bin_edges = model.bin_edges.to(device)
        return model

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
            # rows [old_num:] are randomly init per nn.Embedding default
        self.transformer.wte = new_emb

        # 2) New LM head (tied weights in many HF models)
        old_head = self.lm_head.weight.data  # shape (old_num, emb_dim)
        new_head = nn.Linear(emb_dim, new_num_tokens, bias=False)
        with torch.no_grad():
            new_head.weight[:old_num].copy_(old_head)
        self.lm_head = new_head

        # old_head2 = self.epx_head.weight.data  # shape (old_num, emb_dim)
        # new_head2 = nn.Linear(emb_dim, new_num_tokens, bias=False)
        # with torch.no_grad():
        #     new_head2.weight[:old_num].copy_(old_head2)
        # self.epx_head = new_head2

        # 3) Update config
        self.config.vocab_size = new_num_tokens

        # self.epx_regressor = nn.Sequential(
        #         nn.Linear(self.config.vocab_size, self.config.n_embd),  # map vocab→hidden
        #         nn.ReLU(),
        #         nn.Linear(self.config.n_embd, 1)                   # hidden→scalar
        #     )

    def resize_expression_embeddings(self, new_expression_level: int):
        """
        Grow (or shrink) the expression‐level embedding `wee`
        to support values 0…new_expression_level (inclusive),
        copying old weights and randomly initializing any new rows.
        """
        # how many bins we used to have (including zero)
        old_emb = self.transformer.wee.weight.data
        old_num, emb_dim = old_emb.shape

        # how many we now want (remember wee was built with +1)
        new_num = new_expression_level + 1

        # 1) new embedding
        new_emb = nn.Embedding(new_num, emb_dim)
        with torch.no_grad():
            # copy over the old rows
            new_emb.weight[:old_num].copy_(old_emb)
            # rows [old_num:] are freshly initialized
        self.transformer.wee = new_emb

        # 2) update config so that later calls (or re-initializations) know the new max
        self.config.expression_level = new_expression_level

        self.epx_head = nn.Linear(emb_dim, new_num, bias=False)

        self.epx_regressor = nn.Sequential(
            nn.Linear(new_num, self.config.n_embd),  # map new bin logits → hidden
            nn.ReLU(),
            nn.Linear(self.config.n_embd, 1),
        )

    def forward(
        self,
        idx=None,
        inputs_embeds=None,
        past_key_values=None,
        x_expr=None,
        return_hidden=False,
    ):
        if past_key_values is None:
            past_key_values = [None] * len(self.transformer.h)

        if idx is not None:
            tok_emb = self.transformer.wte(idx)
        else:
            tok_emb = inputs_embeds

        expr_emb = self.transformer.wee(x_expr)
        if self.config.continuous_coords and self.config.coord_token_ids and idx is not None:
            coord_ids = torch.tensor(
                self.config.coord_token_ids, dtype=torch.long, device=idx.device
            )
            coord_mask = (idx.unsqueeze(-1) == coord_ids).any(-1)  # (B, T)
            coord_vals = x_expr.float() / self.config.expression_level
            coord_embs = self.coord_embed(coord_vals.unsqueeze(-1))  # (B, T, n_embd)
            expr_emb = torch.where(coord_mask.unsqueeze(-1), coord_embs, expr_emb)
        x = self.transformer.drop(tok_emb + expr_emb)

        presents_kv = []
        for block, past in zip(self.transformer.h, past_key_values):
            x, kv = block(x, past)
            presents_kv.append(kv)

        x = self.transformer.ln_f(x)

        logits_labels = self.lm_head(x)
        logits_exp_bins = self.epx_head(x)
        logits_exp_real = self.epx_regressor(logits_exp_bins)  # (B, T, 1)

        return logits_labels, logits_exp_bins, logits_exp_real, tuple(presents_kv)

    # @torch.no_grad()
    # def generate_cellGenesis(self, input_ids, expression_level, max_new_tokens, ignore_Idx=None, top_k=None, gamma=1.0, override_gene_sequence=None, override_expr_sequence=None, verbose=False):
    #     output_idx = input_ids
    #     output_expr = expression_level
    #     real_expr = expression_level.float()
    #     past_key_values = None

    #     predicted_gene_tokens = []
    #     predicted_expression_bins = []
    #     predicted_real_expr = []

    #     B = input_ids.size(0)
    #     finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)

    #     first_time = True

    #     while output_idx.shape[1] < max_new_tokens:
    #         if first_time == True:
    #             logits_cls, logits_exp_bins, logits_exp_real, past_key_values = self(idx=input_ids, x_expr=expression_level, past_key_values=past_key_values)
    #             first_time = False
    #         else:
    #             logits_cls, logits_exp_bins, logits_exp_real, past_key_values = self(idx=output_idx[:, -1:], x_expr=output_expr[:, -1:], past_key_values=past_key_values)

    #         logits_cls = logits_cls[:, -1, :]
    #         logits_exp_bins = logits_exp_bins[:, -1, :]
    #         logits_exp_real = logits_exp_real[:, -1].float()

    #         if ignore_Idx is not None:
    #             logits_cls[:, ignore_Idx] = float('-inf')
    #         seen = output_idx[:, :-1]
    #         for b in range(seen.size(0)):
    #             logits_cls[b, seen[b]] = float('-inf')

    #         if top_k is not None:
    #             v, _ = torch.topk(logits_cls, min(top_k, logits_cls.size(-1)), dim=-1)
    #             logits_cls[logits_cls < v[:, [-1]]] = float('-inf')

    #         # probs = F.softmax(logits_cls, dim=-1)
    #         # probs[:, 0] *= gamma
    #         # next_token = torch.multinomial(probs, num_samples=1)

    #         eos_id = 0
    #         eos_threshold = 0.95  # or 0.95
    #         probs = F.softmax(logits_cls, dim=-1)
    #         forced_eos = probs[:, eos_id] >= eos_threshold

    #         # Sample as usual
    #         sampled_tokens = torch.multinomial(probs, num_samples=1)

    #         # Overwrite with EOS where forced
    #         next_token = torch.where(
    #             forced_eos.unsqueeze(1),  # shape (batch_size, 1)
    #             torch.full_like(sampled_tokens, eos_id),
    #             sampled_tokens
    #         )

    #         # Mark items that have generated EOS (0)
    #         newly_finished = (next_token.squeeze(1) == 0)
    #         finished |= newly_finished

    #         next_expr_real = logits_exp_real
    #         bin_next = torch.argmax(logits_exp_bins, dim=-1, keepdim=True)  # (B, 1)

    #         predicted_gene_tokens.append(next_token)
    #         predicted_expression_bins.append(bin_next)
    #         predicted_real_expr.append(next_expr_real)

    #         step_idx = output_idx.size(1)

    #         if override_gene_sequence is not None and step_idx < override_gene_sequence.size(1):
    #             next_token_input = override_gene_sequence[:, step_idx].unsqueeze(1)
    #         else:
    #             next_token_input = next_token

    #         if override_expr_sequence is not None and step_idx < override_expr_sequence.size(1):
    #             next_expr_input = override_expr_sequence[:, step_idx].unsqueeze(1)
    #         else:
    #             next_expr_input = bin_next

    #         output_idx = torch.cat([output_idx, next_token_input], dim=1)
    #         output_expr = torch.cat([output_expr, next_expr_input], dim=1)
    #         real_expr = torch.cat([real_expr, next_expr_real], dim=1)

    #         if finished.all():
    #             break

    #     predicted_gene_tokens = torch.cat(predicted_gene_tokens, dim=1)
    #     predicted_expression_bins = torch.cat(predicted_expression_bins, dim=1)
    #     predicted_real_expr = torch.cat(predicted_real_expr, dim=1)

    #     return predicted_gene_tokens, predicted_expression_bins, predicted_real_expr

    @torch.no_grad()
    def generate_cellGenesis(
        self,
        input_ids,
        expression_level,
        max_new_tokens,
        ignore_Idx=None,
        top_k=None,
        gamma=1.0,
        override_gene_sequence=None,
        override_expr_sequence=None,
        verbose=False,
        select_max_index_margin=0.01,
    ):
        """
        Behavior is identical to the original version:
          - Same sampling distribution (top-k masking, then softmax over full vocab)
          - Same forced-EOS trigger using probs[:, eos_id] >= eos_threshold
          - Same override logic
          - Same returned tensors

        Speedups:
          - Preallocated outputs (no per-step torch.cat O(T^2) copies)
          - Vectorized 'seen token' masking (no Python loop over batch)
          - KV cache usage pattern unchanged
        """
        device = input_ids.device
        B = input_ids.size(0)
        T0 = input_ids.size(1)
        Tmax = T0 + max_new_tokens

        # Track real-valued expression in float
        real_expr = expression_level.float()

        # Preallocate outputs (zeros are fine; matches original behavior)
        out_idx = input_ids.new_full((B, Tmax), 0)
        out_expr = expression_level.new_full((B, Tmax), 0)
        out_real = real_expr.new_zeros((B, Tmax))  # (B, T) float

        # Copy the prefix once
        out_idx[:, :T0] = input_ids
        out_expr[:, :T0] = expression_level
        out_real[:, :T0] = real_expr

        pred_gene_steps = []  # list of (B,1)
        pred_bin_steps = []  # list of (B,1)
        pred_real_steps = []  # list of (B,) values per step (we'll stack to (B, L))

        finished = torch.zeros(B, dtype=torch.bool, device=device)
        first_time = True
        write_pos = T0
        past_key_values = None

        eos_id = 0
        eos_threshold = 0.95  # same as your original

        # --- no-retrain bin-decode experiment ----------------------------------
        # How the per-step expression BIN (recorded + fed back via wee) is chosen.
        # Default "argmax" reproduces the original behavior byte-for-byte. "expected"
        # uses the posterior-mean bin (robust to a flat/zero-dominated bin head, which
        # is what collapses under autoregressive feedback); "sample" draws from the
        # tempered categorical. Controlled by env vars so no signatures change.
        _bin_mode = os.environ.get("MIMYR_BIN_DECODE", "argmax")
        _bin_temp = float(os.environ.get("MIMYR_BIN_TEMP", "1.0"))

        while write_pos < Tmax:
            # Forward pass
            if first_time:
                logits_cls, logits_exp_bins, logits_exp_real, past_key_values = self(
                    idx=out_idx[:, :write_pos],
                    x_expr=out_expr[:, :write_pos],
                    past_key_values=None,
                )
                first_time = False
            else:
                logits_cls, logits_exp_bins, logits_exp_real, past_key_values = self(
                    idx=out_idx[:, write_pos - 1 : write_pos],
                    x_expr=out_expr[:, write_pos - 1 : write_pos],
                    past_key_values=past_key_values,
                )

            # Take last-step logits
            logits_cls_step = logits_cls[:, -1, :]  # (B, V)
            logits_bins_step = logits_exp_bins[:, -1, :]  # (B, E)

            # --- Robust shape handling for real-valued head ---
            # Accept (B,), (B,1), or (B,1,1) and normalize to:
            #   - next_expr_real_vec: (B,) for storage/return
            #   - next_expr_real_col: (B,1) for writing into out_real[:, write_pos:write_pos+1]
            logits_real_step = logits_exp_real[
                :, -1
            ]  # could be (B,), (B,1), or (B,1,1)
            # Squeeze all trailing singleton dims to get (B,) if possible
            next_expr_real_vec = logits_real_step.squeeze()  # target (B,)
            if next_expr_real_vec.dim() != 1:
                # If it's still not (B,), force it
                next_expr_real_vec = next_expr_real_vec.view(B)
            next_expr_real_col = next_expr_real_vec.unsqueeze(1)  # (B,1)

            # Apply ignore list and no-repeat (vectorized)
            if ignore_Idx is not None:
                logits_cls_step[:, ignore_Idx] = float("-inf")

            seen = out_idx[:, :write_pos]  # (B, T_prev)
            seen_mask = torch.zeros_like(logits_cls_step, dtype=torch.bool)  # (B, V)
            seen = seen.clamp_min(0)
            seen_mask.scatter_(1, seen, True)
            seen_mask[:, eos_id] = False  # padding zeros must not block EOS termination
            logits_cls_step = logits_cls_step.masked_fill(seen_mask, float("-inf"))

            # Optional top-k -> identical semantics as original (mask non-topk to -inf)
            if top_k is not None:
                k = min(top_k, logits_cls_step.size(-1))
                v, _ = torch.topk(logits_cls_step, k, dim=-1)
                kth = v[:, [-1]]
                neg_inf = logits_cls_step.new_full((), float("-inf"))
                logits_cls_step = torch.where(
                    logits_cls_step >= kth, logits_cls_step, neg_inf
                )

            # mask tokens with index >= previous token (enforce strictly decreasing order)
            prev_tokens = out_idx[:, write_pos - 1]  # (B,)
            token_range = torch.arange(logits_cls_step.size(-1), device=logits_cls_step.device).unsqueeze(0)
            logits_cls_step = logits_cls_step.masked_fill(token_range >= prev_tokens.unsqueeze(1), float("-inf"))

            # Softmax and sample over the (masked) full vocab
            probs = F.softmax(logits_cls_step, dim=-1)
            forced_eos = probs[:, eos_id] >= eos_threshold

            if select_max_index_margin > 0.0:
                # Among still-allowed (finite) tokens whose probability is within
                # `margin` of the top probability, pick the HIGHEST token index.
                # The argmax is always within the margin (diff 0), so this returns
                # the argmax unless a higher-index token is also within the margin —
                # it never falls back to sampling.
                max_prob = probs.max(dim=-1, keepdim=True).values  # (B,1)
                eligible = ((max_prob - probs) < select_max_index_margin) & torch.isfinite(logits_cls_step)  # (B,V)
                token_indices = torch.arange(
                    logits_cls_step.size(-1), device=logits_cls_step.device
                ).unsqueeze(0).expand_as(logits_cls_step)
                next_token_raw = token_indices.masked_fill(~eligible, -1).max(dim=-1, keepdim=True).values  # (B,1)
                next_token = torch.where(
                    forced_eos.unsqueeze(1),
                    next_token_raw.new_full((B, 1), eos_id),
                    next_token_raw,
                )
            else:
                sampled_tokens = torch.multinomial(probs, num_samples=1)  # (B,1)
                next_token = torch.where(
                    forced_eos.unsqueeze(1),
                    sampled_tokens.new_full((B, 1), eos_id),
                    sampled_tokens,
                )

            # Binned expression for this step
            if _bin_mode == "argmax":
                bin_next = torch.argmax(logits_bins_step, dim=-1, keepdim=True)  # (B,1)
            else:
                probs_bin = F.softmax(logits_bins_step / _bin_temp, dim=-1)  # (B,E)
                if _bin_mode == "expected":
                    levels = torch.arange(
                        logits_bins_step.size(-1),
                        device=logits_bins_step.device,
                        dtype=probs_bin.dtype,
                    )
                    exp_bin = (probs_bin * levels).sum(dim=-1, keepdim=True)  # (B,1) float
                    bin_next = exp_bin.round().long().clamp_(0, logits_bins_step.size(-1) - 1)
                elif _bin_mode == "sample":
                    bin_next = torch.multinomial(probs_bin, num_samples=1)  # (B,1)
                else:
                    raise ValueError(f"unknown MIMYR_BIN_DECODE={_bin_mode!r}")

            # Accumulate step outputs (same shapes as before)
            pred_gene_steps.append(next_token)  # (B,1)
            pred_bin_steps.append(bin_next)  # (B,1)
            pred_real_steps.append(next_expr_real_vec)  # (B,)

            # Overrides (unchanged behavior)
            step_idx = write_pos
            if (
                override_gene_sequence is not None
                and step_idx < override_gene_sequence.size(1)
            ):
                next_token_input = override_gene_sequence[:, step_idx : step_idx + 1]
            else:
                next_token_input = next_token

            if (
                override_expr_sequence is not None
                and step_idx < override_expr_sequence.size(1)
            ):
                next_expr_input = override_expr_sequence[:, step_idx : step_idx + 1]
            else:
                next_expr_input = bin_next

            # Write into preallocated buffers
            out_idx[:, write_pos : write_pos + 1] = next_token_input
            out_expr[:, write_pos : write_pos + 1] = next_expr_input
            out_real[:, write_pos : write_pos + 1] = next_expr_real_col  # (B,1)

            # Finished tracking identical to original (EOS == 0)
            newly_finished = next_token.squeeze(1) == eos_id
            finished |= newly_finished

            write_pos += 1
            if finished.all():
                break

        # Concatenate outputs exactly like the original
        predicted_gene_tokens = torch.cat(pred_gene_steps, dim=1)  # (B, L)
        predicted_expression_bins = torch.cat(pred_bin_steps, dim=1)  # (B, L)
        predicted_real_expr = torch.stack(pred_real_steps, dim=1)  # (B, L)

        return predicted_gene_tokens, predicted_expression_bins, predicted_real_expr

    @torch.no_grad()
    def generate_cellGenesis_fast(
        self,
        input_ids,
        expression_level,
        max_new_tokens,
        ignore_Idx=None,
        top_k=None,
        gamma=1.0,
        override_gene_sequence=None,
        override_expr_sequence=None,
        verbose=False,
        select_max_index_margin=0.01,
    ):
        """
        Pruned decoding with correct KV-cache indexing.
        - Advances only unfinished rows.
        - Maintains an 'alive' mapping of original batch indices currently in the cache.
        - Scatters per-step results back to full (B, 1) tensors, so returns are (B, L).

        Sampling semantics match your original (mask -> softmax over full vocab -> multinomial -> forced EOS).
        """

        device = input_ids.device
        B = input_ids.size(0)
        T0 = input_ids.size(1)
        Tmax = T0 + max_new_tokens

        # Preallocate rolling buffers for bookkeeping (full batch)
        real_expr_prefix = expression_level.float()
        out_idx = input_ids.new_full((B, Tmax), 0)
        out_expr = expression_level.new_full((B, Tmax), 0)
        out_real = real_expr_prefix.new_zeros((B, Tmax))
        out_idx[:, :T0] = input_ids
        out_expr[:, :T0] = expression_level
        out_real[:, :T0] = real_expr_prefix

        # Per-step full-batch outputs (we'll cat at the end)
        step_tokens_full, step_bins_full, step_real_full = [], [], []

        # Global finished flags (size B)
        finished_global = torch.zeros(B, dtype=torch.bool, device=device)
        write_pos = T0
        eos_id = 0
        eos_threshold = 0.95

        # No-retrain bin-decode experiment (see generate_cellGenesis). This is the
        # fast path actually used at inference (Mimyr.generate ... fast=True). Default
        # "argmax" is byte-identical to original; "expected"/"sample" change how the
        # per-step bin (recorded + fed back via wee) is chosen.
        _bin_mode = os.environ.get("MIMYR_BIN_DECODE", "argmax")
        _bin_temp = float(os.environ.get("MIMYR_BIN_TEMP", "1.0"))

        # Start with ALL rows in the cache, in original order
        alive = torch.arange(
            B, device=device, dtype=torch.long
        )  # maps cache rows -> original batch idx

        # Build initial KV cache from the full prefix
        _, _, _, past_key_values = self(
            idx=out_idx[:, :write_pos],
            x_expr=out_expr[:, :write_pos],
            past_key_values=None,
        )

        # Prepare ignore mask builder (tied to vocab size)
        ignore_mask = None

        def _build_ignore_mask(V):
            nonlocal ignore_mask
            if ignore_Idx is None:
                ignore_mask = None
                return
            if isinstance(ignore_Idx, int):
                idx = torch.tensor([ignore_Idx], device=device, dtype=torch.long)
            elif isinstance(ignore_Idx, (list, tuple)):
                idx = torch.tensor(ignore_Idx, device=device, dtype=torch.long)
            else:
                idx = ignore_Idx.to(device=device, dtype=torch.long)
            idx = idx.clamp(min=0, max=V - 1)
            ignore_mask = torch.zeros((1, V), dtype=torch.bool, device=device)
            ignore_mask.scatter_(1, idx.view(1, -1), True)

        while write_pos < Tmax:
            # Finished flags relative to current cache (alive)
            finished_rel = finished_global[alive]  # (Ba,)
            rel_active = (
                (~finished_rel).nonzero(as_tuple=False).squeeze(1)
            )  # indices into cache rows
            if rel_active.numel() == 0:
                break  # all done

            # Global indices for active rows (for scatter back)
            global_idx = alive[rel_active]  # (Ba,)

            # One-step inputs for active rows
            last_tok_active = out_idx[global_idx, write_pos - 1 : write_pos]  # (Ba,1)
            last_expr_active = out_expr[global_idx, write_pos - 1 : write_pos]  # (Ba,1)

            # Slice PKV by RELATIVE indices (cache is currently ordered by 'alive')
            pkv_active = _pkv_index_select(past_key_values, rel_active)

            # One-step forward
            logits_cls_a, logits_bins_a, logits_real_a, pkv_next = self(
                idx=last_tok_active, x_expr=last_expr_active, past_key_values=pkv_active
            )
            # After this step, the cache should ONLY contain the active rows in the same order
            past_key_values = pkv_next
            # And 'alive' should also be reduced to only those active rows (we'll drop finished after sampling)
            alive_active = global_idx.clone()  # current active's global ids

            # Extract last-step logits for active rows
            logits_cls_step = logits_cls_a[:, -1, :]  # (Ba, V)
            logits_bins_step = logits_bins_a[:, -1, :]  # (Ba, E)
            V = logits_cls_step.size(1)

            # Build/refresh ignore mask when V changes
            if ignore_mask is None or ignore_mask.size(1) != V:
                _build_ignore_mask(V)

            # Real-valued head: normalize to (Ba,) and (Ba,1)
            # lr = logits_real_a[:, -1]
            # next_real_vec = lr.squeeze()
            # if next_real_vec.dim() != 1:
            #     next_real_vec = next_real_vec.view(next_real_vec.size(0))
            # next_real_col = next_real_vec.unsqueeze(1)          # (Ba,1)
            Ba = logits_real_a.size(0)  # active batch size
            lr = logits_real_a[:, -1]  # could be (Ba,), (Ba,1), or (Ba,1,1)
            # Flatten trailing dims, keep first element per row → (Ba,)
            next_real_vec = lr.reshape(Ba, -1)[:, 0].contiguous()
            next_real_col = next_real_vec.unsqueeze(1)  # (Ba,1)

            # Apply ignore mask
            if ignore_mask is not None:
                logits_cls_step = logits_cls_step.masked_fill(
                    ignore_mask, float("-inf")
                )

            # No-repeat mask (use SEEN TOKENS from GLOBAL buffers)
            seen_active = out_idx[alive_active, :write_pos]  # (Ba, Tprev)
            seen_active = seen_active.clamp(min=0, max=V - 1).to(dtype=torch.long)
            seen_mask = torch.zeros_like(logits_cls_step, dtype=torch.bool)  # (Ba, V)
            seen_mask.scatter_(1, seen_active, True)
            seen_mask[:, eos_id] = False  # padding zeros must not block EOS termination
            logits_cls_step = logits_cls_step.masked_fill(seen_mask, float("-inf"))

            # Optional safe top-k
            if top_k is not None:
                logits_cls_step = _safe_topk_mask_(logits_cls_step, top_k)

            # mask tokens with index >= previous token (enforce strictly decreasing order)
            prev_tokens_active = last_tok_active.squeeze(1)  # (Ba,)
            token_range = torch.arange(logits_cls_step.size(-1), device=logits_cls_step.device).unsqueeze(0)
            logits_cls_step = logits_cls_step.masked_fill(token_range >= prev_tokens_active.unsqueeze(1), float("-inf"))

            # Ensure at least one finite logit per row (EOS fallback)
            row_has_finite = torch.isfinite(logits_cls_step).any(dim=1)
            if not row_has_finite.all():
                bad = ~row_has_finite
                logits_cls_step[bad, :] = float("-inf")
                eos_id_safe = 0 if V == 0 else min(max(0, eos_id), V - 1)
                logits_cls_step[bad, eos_id_safe] = 0.0

            # Sample
            probs = F.softmax(logits_cls_step, dim=-1)
            forced_eos = probs[:, eos_id] >= eos_threshold

            if select_max_index_margin > 0.0:
                # Among still-allowed (finite) tokens whose probability is within
                # `margin` of the top probability, pick the HIGHEST token index.
                # The argmax is always within the margin (diff 0), so this returns
                # the argmax unless a higher-index token is also within the margin —
                # it never falls back to sampling.
                max_prob = probs.max(dim=-1, keepdim=True).values  # (Ba,1)
                eligible = ((max_prob - probs) < select_max_index_margin) & torch.isfinite(logits_cls_step)  # (Ba,V)
                token_indices = torch.arange(
                    logits_cls_step.size(-1), device=logits_cls_step.device
                ).unsqueeze(0).expand_as(logits_cls_step)
                next_token_raw = token_indices.masked_fill(~eligible, -1).max(dim=-1, keepdim=True).values  # (Ba,1)
                next_token_active = torch.where(
                    forced_eos.unsqueeze(1),
                    next_token_raw.new_full((next_token_raw.size(0), 1), eos_id),
                    next_token_raw,
                )  # (Ba,1)
            else:
                sampled_tokens = torch.multinomial(probs, num_samples=1)  # (Ba,1)
                next_token_active = torch.where(
                    forced_eos.unsqueeze(1),
                    sampled_tokens.new_full((sampled_tokens.size(0), 1), eos_id),
                    sampled_tokens,
                )  # (Ba,1)

            # Expression-bin for active rows
            if _bin_mode == "argmax":
                bin_next_active = torch.argmax(
                    logits_bins_step, dim=-1, keepdim=True
                )  # (Ba,1)
            else:
                probs_bin = F.softmax(logits_bins_step / _bin_temp, dim=-1)  # (Ba,E)
                if _bin_mode == "expected":
                    levels = torch.arange(
                        logits_bins_step.size(-1),
                        device=logits_bins_step.device,
                        dtype=probs_bin.dtype,
                    )
                    exp_bin = (probs_bin * levels).sum(dim=-1, keepdim=True)  # (Ba,1)
                    bin_next_active = exp_bin.round().long().clamp_(
                        0, logits_bins_step.size(-1) - 1
                    )
                elif _bin_mode == "sample":
                    bin_next_active = torch.multinomial(probs_bin, num_samples=1)  # (Ba,1)
                else:
                    raise ValueError(f"unknown MIMYR_BIN_DECODE={_bin_mode!r}")

            # Build full-batch (B,1) outputs for this step and scatter active results
            tok_full = input_ids.new_full((B, 1), eos_id)
            bin_full = expression_level.new_full((B, 1), 0)
            real_full = out_real.new_zeros((B, 1))
            tok_full[alive_active] = next_token_active
            bin_full[alive_active] = bin_next_active
            real_full[alive_active] = next_real_col
            step_tokens_full.append(tok_full)
            step_bins_full.append(bin_full)
            step_real_full.append(real_full)

            # Overrides (apply to active rows only)
            step_idx = write_pos
            if (
                override_gene_sequence is not None
                and step_idx < override_gene_sequence.size(1)
            ):
                next_token_input_active = override_gene_sequence[
                    alive_active, step_idx : step_idx + 1
                ]
            else:
                next_token_input_active = next_token_active

            if (
                override_expr_sequence is not None
                and step_idx < override_expr_sequence.size(1)
            ):
                next_expr_input_active = override_expr_sequence[
                    alive_active, step_idx : step_idx + 1
                ]
            else:
                next_expr_input_active = bin_next_active

            # Write next tokens/expr into rolling buffers for active rows
            out_idx[alive_active, write_pos : write_pos + 1] = next_token_input_active
            out_expr[alive_active, write_pos : write_pos + 1] = next_expr_input_active
            out_real[alive_active, write_pos : write_pos + 1] = next_real_col

            # Update finished flags globally for active rows
            newly_finished_active = next_token_active.squeeze(1) == eos_id  # (Ba,)
            finished_global[alive_active] |= newly_finished_active

            # Shrink 'alive' and cache to ONLY rows that are still unfinished after this step
            keep_rel = ~newly_finished_active  # (Ba,)
            if keep_rel.any():
                # Reorder cache and 'alive' to only unfinished rows
                keep_rel_idx = keep_rel.nonzero(as_tuple=False).squeeze(
                    1
                )  # rel indices to keep
                past_key_values = _pkv_index_select(past_key_values, keep_rel_idx)
                alive = alive_active[keep_rel_idx]  # update mapping
            else:
                # No rows left in cache
                alive = alive_active.new_empty((0,), dtype=torch.long)
                past_key_values = _pkv_index_select(
                    past_key_values, keep_rel.nonzero(as_tuple=False).squeeze(1)
                )  # empty select

            write_pos += 1
            if finished_global.all():
                break

        # Concatenate step outputs to (B, L)
        if step_tokens_full:
            predicted_gene_tokens = torch.cat(step_tokens_full, dim=1)
            predicted_expression_bins = torch.cat(step_bins_full, dim=1)
            predicted_real_expr = torch.cat(step_real_full, dim=1)
        else:
            predicted_gene_tokens = input_ids.new_zeros((B, 0), dtype=input_ids.dtype)
            predicted_expression_bins = expression_level.new_zeros(
                (B, 0), dtype=expression_level.dtype
            )
            predicted_real_expr = out_real.new_zeros((B, 0))

        return predicted_gene_tokens, predicted_expression_bins, predicted_real_expr


def _safe_topk_mask_(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    In-place-ish mask: keep top-k per row, set others to -inf.
    Handles k<=0, k>=V, and NaNs.
    """
    if k is None:
        return logits
    V = logits.size(-1)
    try:
        k = int(k)
    except Exception:
        return logits
    if k <= 0 or k >= V:
        return logits
    # Replace NaNs with -inf to make ordering defined
    neg_inf = logits.new_full((), float("-inf"))
    logits = torch.where(torch.isnan(logits), neg_inf, logits)
    v, _ = torch.topk(logits, k, dim=-1)  # (B, k)
    kth = v[:, [-1]]  # (B, 1)
    logits.masked_fill_(logits < kth, neg_inf)
    return logits


def _pkv_index_select(past_key_values, rel_index):
    """
    Select batch dim (dim=0) of KV cache by RELATIVE indices.
    Assumes PKV structure: list of (k, v) per layer with shapes (Ba, ...).
    """
    if past_key_values is None:
        return None
    # Fast path: selecting all rows — alive is already pruned to unfinished rows,
    # so rel_index is always [0..N-1] here; skip the full copy.
    if rel_index.numel() == past_key_values[0][0].size(0):
        return past_key_values
    new_pkv = []
    for k, v in past_key_values:
        new_k = k.index_select(0, rel_index)
        new_v = v.index_select(0, rel_index)
        new_pkv.append((new_k, new_v))
    return new_pkv


@dataclass
class SampleDecoderOutput(SampleDecoderOnlyOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    expression: Optional[torch.LongTensor] = None
