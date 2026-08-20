"""
Configuración central del proyecto.

Aquí vive TODO lo que define el tamaño y comportamiento de tu modelo.
Cambiar estos números es cómo "haces crecer" tu IA con el tiempo.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # --- Tamaño del modelo ---
    vocab_size: int = 8000       # tamaño del vocabulario del tokenizador BPE
    block_size: int = 256        # longitud máxima de contexto (tokens que "recuerda" a la vez)
    n_layer: int = 6             # número de bloques transformer (profundidad)
    n_head: int = 6              # número de cabezas de atención
    n_embd: int = 384            # dimensión del embedding (ancho del modelo)
    dropout: float = 0.1
    bias: bool = True            # usar bias en Linear/LayerNorm (True = más fiel a GPT-2)

    # Referencia de escala (para que entiendas dónde estás parado):
    #   Esta config por defecto  -> ~10-15M parámetros (cabe en Colab gratis, entrena en horas)
    #   GPT-2 small               -> 124M parámetros
    #   GPT-2 XL                  -> 1.5B parámetros
    #   Modelos "grandes" actuales -> decenas/cientos de miles de millones


@dataclass
class TrainConfig:
    # --- Datos ---
    data_dir: str = "data"
    out_dir: str = "checkpoints"

    # --- Entrenamiento ---
    batch_size: int = 32
    grad_accum_steps: int = 4     # simula un batch más grande sin usar más memoria
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # --- Sistema ---
    device: str = "cuda"          # se cambia solo a "cpu" si no hay GPU disponible
    dtype: str = "bfloat16"       # "float32" si tu GPU no soporta bfloat16
    compile_model: bool = True    # torch.compile acelera el entrenamiento (PyTorch 2.x)
    seed: int = 1337
