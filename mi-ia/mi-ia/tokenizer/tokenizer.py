"""
Wrapper delgado sobre el tokenizador BPE ya entrenado (tokenizer.json).
Lo usan tanto data/prepare_data.py como inference/generate.py.
"""

import os
from tokenizers import Tokenizer


class MiTokenizer:
    def __init__(self, path: str = None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No existe {path}. Primero corre tokenizer/train_tokenizer.py"
            )
        self.tok = Tokenizer.from_file(path)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()
