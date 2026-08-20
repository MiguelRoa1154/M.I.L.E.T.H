"""
Entrena un tokenizador BPE (Byte-Pair Encoding) DESDE CERO sobre tu corpus.

BPE es el mismo tipo de tokenizador que usan GPT-2/3/4: empieza con
caracteres individuales y va fusionando los pares más frecuentes hasta
formar un vocabulario de subpalabras.

Uso:
    python tokenizer/train_tokenizer.py --input data/raw.txt --vocab_size 8000
"""

import argparse
import os

from tokenizers import ByteLevelBPETokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw.txt",
                         help="Ruta al archivo de texto (tu corpus)")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--out_dir", type=str, default="tokenizer")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"No encontré '{args.input}'. Pon tu corpus de texto ahí primero "
            f"(puede ser cualquier .txt en español, tamaño libre)."
        )

    print(f"Entrenando tokenizador BPE con vocab_size={args.vocab_size} ...")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[args.input],
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<|pad|>"],
    )

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.out_dir, "tokenizer.json"))

    print(f"Tokenizador guardado en {args.out_dir}/tokenizer.json")
    print("Prueba rápida:")
    sample = "Esto es una prueba de mi propio tokenizador."
    ids = tokenizer.encode(sample).ids
    print(f"  Texto:  {sample}")
    print(f"  Tokens: {ids}")
    print(f"  Decodificado: {tokenizer.decode(ids)}")


if __name__ == "__main__":
    main()
