"""
Genera texto usando un checkpoint ya entrenado. Así "hablas" con tu modelo.

Uso:
    python inference/generate.py --prompt "Había una vez" --max_new_tokens 200
"""

import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt import GPT
from tokenizer.tokenizer import MiTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Había una vez")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8,
                         help="más alto = más creativo/aleatorio; más bajo = más conservador")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ckpt.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Cargando checkpoint {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = ckpt["model_config"]

    model = GPT(model_cfg).to(device)
    state_dict = ckpt["model"]
    # limpia el prefijo que agrega torch.compile si estaba activo
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    tok = MiTokenizer()

    ids = tok.encode(args.prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    print(f"\nPrompt: {args.prompt}")
    print("Generando...\n")

    with torch.no_grad():
        y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)

    output = tok.decode(y[0].tolist())
    print(output)


if __name__ == "__main__":
    main()
