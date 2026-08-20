[README.md](https://github.com/user-attachments/files/31272890/README.md)
# Mi IA — Transformer entrenado desde cero

Proyecto propio de un modelo de lenguaje (arquitectura tipo GPT) construido
desde cero: tokenizador, arquitectura, entrenamiento e inferencia. Pensado
para crecer por etapas, empezando en Google Colab (GPU gratis).

## Filosofía del proyecto

Esto NO es un modelo del tamaño de GPT-4/Claude (eso requiere millones de
dólares en cómputo). Es un transformer real y funcional, más pequeño, que
tú entiendes y controlas por completo, y que puedes ir haciendo crecer:
más datos, más parámetros, más entrenamiento, mejores capacidades.

## Estructura

```
mi-ia/
├── config/            # hiperparámetros del modelo y del entrenamiento
│   └── config.py
├── data/              # datasets crudos y tokenizados (.bin)
│   └── prepare_data.py
├── tokenizer/         # tokenizador BPE propio
│   ├── train_tokenizer.py
│   └── tokenizer.json      (se genera al entrenar)
├── model/             # arquitectura del transformer (desde cero, PyTorch)
│   └── gpt.py
├── training/          # loop de entrenamiento
│   └── train.py
├── inference/         # generación de texto con el modelo entrenado
│   └── generate.py
├── checkpoints/        # pesos guardados del modelo (.pt)
└── notebooks/
    └── Entrenar_en_Colab.ipynb   # todo listo para correr en Colab
```

## Flujo de trabajo (etapas)

1. **Preparar datos**: pones tu corpus de texto en `data/raw.txt`
   (español, el idioma/dominio que quieras) y corres `prepare_data.py`.
2. **Entrenar el tokenizador**: `tokenizer/train_tokenizer.py` aprende
   un vocabulario BPE (Byte-Pair Encoding) a partir de tu corpus.
3. **Entrenar el modelo**: `training/train.py` entrena el transformer
   desde cero sobre los datos tokenizados. Guarda checkpoints.
4. **Generar texto**: `inference/generate.py` carga un checkpoint y
   genera texto — así "hablas" con tu modelo.

## Cómo empezar (rápido, en Colab)

1. Sube este repo a GitHub (o descarga el .zip y súbelo directo a Colab).
2. Abre `notebooks/Entrenar_en_Colab.ipynb` en Google Colab.
3. Activa GPU: `Entorno de ejecución > Cambiar tipo de entorno > GPU (T4)`.
4. Corre las celdas en orden. Todo — instalación, datos, tokenizador,
   entrenamiento, generación — está automatizado ahí.

## Cómo empezar (local, con tu propia GPU)

```bash
pip install -r requirements.txt
python tokenizer/train_tokenizer.py
python data/prepare_data.py
python training/train.py
python inference/generate.py --prompt "Había una vez"
```

## Próximos pasos para hacerlo crecer

- Aumentar `n_layer`, `n_head`, `n_embd` en `config/config.py` (modelo más grande)
- Usar un corpus más grande (Wikipedia en español, OSCAR, libros de dominio público)
- Fine-tuning para una tarea específica (chat, código, un estilo particular)
- Instrucción-tuning: entrenar con pares pregunta/respuesta para que "obedezca"
- (Avanzado) Un mini-RLHF: enseñarle a preferir ciertas respuestas sobre otras
