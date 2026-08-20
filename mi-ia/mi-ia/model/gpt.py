"""
Arquitectura de un modelo de lenguaje tipo GPT, escrita DESDE CERO con
PyTorch puro (solo usamos nn.Linear, nn.Embedding, etc. como bloques
básicos — la lógica de atención y del transformer la escribimos nosotros).

Piezas:
  1. CausalSelfAttention  -> el mecanismo de "atención" (el corazón del transformer)
  2. MLP                  -> red feed-forward de cada bloque
  3. Block                -> un bloque transformer completo (atención + MLP + residuales)
  4. GPT                  -> el modelo completo (embeddings + N bloques + cabeza de salida)
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """
    Self-attention "causal": cada token solo puede atender a los tokens
    anteriores (y a sí mismo), nunca a los futuros. Esto es lo que permite
    entrenar el modelo a predecir "la siguiente palabra".
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # una sola matriz que produce Q, K, V juntos (más eficiente)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # proyección de salida tras combinar las cabezas
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # máscara causal: matriz triangular para bloquear "ver el futuro"
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, secuencia, embedding

        # Q, K, V para todas las cabezas a la vez
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # atención: qué tanto "presta atención" cada token a los anteriores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, hd) — promedio ponderado de los "values"
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # recombinar cabezas

        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Red feed-forward simple: expande, activa (GELU), y comprime de vuelta."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Un bloque transformer completo:
      x = x + Atención(LayerNorm(x))
      x = x + MLP(LayerNorm(x))
    Las conexiones residuales (x + ...) son clave para poder entrenar
    redes profundas sin que el gradiente se "desvanezca".
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """El modelo completo: embeddings de token + posición, N bloques, cabeza final."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),   # embedding de tokens
                wpe=nn.Embedding(config.block_size, config.n_embd),   # embedding posicional
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        # cabeza de salida: proyecta de vuelta al tamaño del vocabulario
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: compartir pesos entre el embedding de entrada y la
        # cabeza de salida (truco estándar de GPT-2, reduce parámetros)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Modelo inicializado: {n_params/1e6:.2f}M parámetros")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Secuencia de longitud {T} excede block_size={self.config.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)   # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Genera texto token por token, autoregresivamente."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        Separa los parámetros: a las matrices de pesos SÍ se les aplica
        weight decay (regularización); a los bias y LayerNorm NO (es lo
        estándar, mejora el entrenamiento).
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)

        optim_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
