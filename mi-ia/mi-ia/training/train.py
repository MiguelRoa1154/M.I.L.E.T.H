"""
Loop de entrenamiento del modelo.

Uso:
    python training/train.py
"""

import math
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import ModelConfig, TrainConfig
from model.gpt import GPT


def get_batch(split, data_dir, block_size, batch_size, device):
    path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data_dir, block_size, batch_size, device, eval_iters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, data_dir, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it, cfg: TrainConfig):
    """Learning rate con warmup lineal + decaimiento coseno."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if it > cfg.max_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def main():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    if train_cfg.device == "cuda" and not torch.cuda.is_available():
        print("No hay GPU disponible, usando CPU (será mucho más lento).")
        train_cfg.device = "cpu"
        train_cfg.compile_model = False

    torch.manual_seed(train_cfg.seed)
    os.makedirs(train_cfg.out_dir, exist_ok=True)

    print("Configuración del modelo:", model_cfg)
    model = GPT(model_cfg).to(train_cfg.device)

    if train_cfg.compile_model:
        print("Compilando modelo con torch.compile (puede tardar un momento)...")
        model = torch.compile(model)

    optimizer = model.configure_optimizers(
        train_cfg.weight_decay, train_cfg.learning_rate, betas=(0.9, 0.95)
    )

    dtype = torch.bfloat16 if train_cfg.dtype == "bfloat16" else torch.float32
    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if train_cfg.device == "cuda" else torch.no_grad().__class__()

    best_val_loss = float("inf")
    t0 = time.time()

    for it in range(train_cfg.max_iters + 1):
        lr = get_lr(it, train_cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if it % train_cfg.eval_interval == 0:
            losses = estimate_loss(
                model, train_cfg.data_dir, model_cfg.block_size,
                train_cfg.batch_size, train_cfg.device, train_cfg.eval_iters,
            )
            dt = time.time() - t0
            print(f"iter {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, {dt:.1f}s")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                ckpt_path = os.path.join(train_cfg.out_dir, "ckpt.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_config": model_cfg,
                        "iter": it,
                        "val_loss": best_val_loss,
                    },
                    ckpt_path,
                )
                print(f"  -> checkpoint guardado en {ckpt_path} (val_loss={best_val_loss:.4f})")

        # entrenamiento con acumulación de gradiente
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(train_cfg.grad_accum_steps):
            X, Y = get_batch("train", train_cfg.data_dir, model_cfg.block_size,
                              train_cfg.batch_size, train_cfg.device)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / train_cfg.grad_accum_steps
            loss.backward()

        if train_cfg.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        optimizer.step()

    print("Entrenamiento terminado.")


if __name__ == "__main__":
    main()
