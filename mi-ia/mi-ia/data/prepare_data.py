"""
Convierte data/raw.txt en arrays de tokens (train.bin / val.bin) listos
para entrenar. Usa el tokenizador que ya entrenaste en tokenizer/train_tokenizer.py.

Uso:
    python data/prepare_data.py
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.tokenizer import MiTokenizer


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(data_dir, "raw.txt")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"No encontré {raw_path}. Coloca ahí tu corpus de texto (.txt) "
            f"antes de continuar."
        )

    print("Cargando tokenizador...")
    tok = MiTokenizer()

    print(f"Leyendo {raw_path} ...")
    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Tokenizando {len(text):,} caracteres ...")
    ids = tok.encode(text)
    print(f"Total de tokens: {len(ids):,}")

    # split 90% train / 10% validación
    n = len(ids)
    train_ids = ids[: int(n * 0.9)]
    val_ids = ids[int(n * 0.9):]

    train_arr = np.array(train_ids, dtype=np.uint16)
    val_arr = np.array(val_ids, dtype=np.uint16)

    train_arr.tofile(os.path.join(data_dir, "train.bin"))
    val_arr.tofile(os.path.join(data_dir, "val.bin"))

    print(f"train.bin: {len(train_arr):,} tokens")
    print(f"val.bin:   {len(val_arr):,} tokens")
    print("Listo. Ya puedes correr training/train.py")


if __name__ == "__main__":
    main()
