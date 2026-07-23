# Emotion Classification with Attention

End-to-end multi-class emotion classification on the **GoEmotions** dataset,
comparing classical recurrent architectures (LSTM, GRU), a **custom
BiLSTM + Attention** model built from scratch, and a **fine-tuned
DistilBERT** Transformer.

**Target emotions:** Joy · Sadness · Anger · Fear · Surprise · Disgust

---

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Results](#results)
- [Attention Visualization](#attention-visualization)
- [Gradio Demo](#gradio-demo)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

This project implements a complete NLP pipeline from raw data to a
deployable interactive demo:

1. **Data exploration** — class balance, text length, vocabulary, word clouds
2. **Preprocessing** — cleaning, stratified splitting, vocabulary/tokenization
3. **GloVe embeddings** — pretrained `glove.6B.100d` vectors, OOV analysis
4. **Four models** — LSTM, GRU, BiLSTM+Attention, DistilBERT
5. **Evaluation** — confusion matrices, classification reports, comparison table
6. **Interpretability** — custom attention-weight visualization
7. **Deployment** — an interactive Gradio app serving all four models

The full pipeline is implemented in a single Jupyter notebook:
[`notebooks/Emotion_Classification_with_Attention.ipynb`](notebooks/Emotion_Classification_with_Attention.ipynb),
built section-by-section with no placeholders — every cell is executable.

## Models

| Model | Architecture | Notes |
|---|---|---|
| **LSTM** | `Embedding (GloVe) → LSTM → Dropout → Linear` | Baseline recurrent model |
| **GRU** | `Embedding (GloVe) → GRU → Dropout → Linear` | Matched hyperparameters vs. LSTM |
| **BiLSTM + Attention** | `Embedding (GloVe) → BiLSTM → Additive Attention → Dropout → Linear` | Custom attention layer implemented from scratch (Bahdanau-style) |
| **DistilBERT** | `distilbert-base-uncased` fine-tuned end-to-end | Hugging Face `Trainer`, class-weighted loss |

All models share identical: train/val/test splits, class-weighted
cross-entropy loss, and evaluation metrics — for a fair, apples-to-apples
comparison.

## Dataset

[GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions)
(Google Research) — 58k Reddit comments, originally multi-labeled across 27
fine-grained emotions + neutral. This project filters to a clean
**single-label 6-class subset** (Joy, Sadness, Anger, Fear, Surprise,
Disgust), keeping only comments annotated with exactly one of these six
labels, to avoid ambiguous multi-label supervision.

## Project Structure

```
emotion-classification-attention/
├── notebooks/
│   └── Emotion_Classification_with_Attention.ipynb   # full pipeline, section by section
├── data/            # generated at runtime: cleaned/split CSVs, vocab, GloVe matrix, cached tensors
├── models/          # generated at runtime: best checkpoints for all 4 models
├── figures/         # generated at runtime: EDA plots, learning curves, confusion matrices, attention viz
├── reports/         # generated at runtime: EDA/GloVe/class-weight reports, classification reports, comparison table
└── README.md
```

`data/`, `models/`, `figures/`, and `reports/` are populated when you run
the notebook (they map to `/kaggle/working/{data,models,figures,reports}`
on Kaggle) — they are not pre-populated in this repo to keep it lightweight.

## Setup & Usage

**Recommended environment:** Kaggle Notebook, GPU **T4**, **Internet ON**.

1. Create a new Kaggle notebook and import
   `notebooks/Emotion_Classification_with_Attention.ipynb`
   (**File → Import Notebook**).
2. In **Settings**: set **Accelerator → GPU T4**, **Internet → ON**.
3. *(Optional but recommended)* Add the Kaggle dataset containing
   `glove.6B.100d.txt` via **Add Input** to skip re-downloading GloVe from
   Stanford NLP each run.
4. Run the first cell (dependency install), then **restart the session**
   (Run → Restart Session) so the pinned package versions load cleanly.
5. **Run All** — the notebook executes top-to-bottom, section by section,
   saving every artifact (data, models, figures, reports) as it goes.

To run locally instead of on Kaggle, install the dependencies listed in
Section 1.1 of the notebook and adjust `PROJECT_ROOT` from `/kaggle/working`
to a local path.

## Results

The full model comparison table, confusion matrices, and per-class
classification reports are generated in **Section 8** of the notebook and
saved to:

- `reports/model_comparison_table.csv` — Accuracy / Macro-F1 / Weighted-F1 / Precision / Recall per model
- `reports/classification_reports.json` — full per-class precision/recall/F1
- `figures/13_confusion_matrices_all_models.png` — normalized confusion matrices, all 4 models
- `figures/14_model_comparison_bar_chart.png` — grouped metric comparison
- `figures/15_per_class_f1_heatmap.png` — per-emotion F1 across models

*(Populate this section with your own run's numbers after executing the
notebook.)*

## Attention Visualization

The custom `AdditiveAttention` layer (Section 6) makes the BiLSTM+Attention
model's decisions inspectable. Section 9 extracts and visualizes per-token
attention weights:

- `figures/17_attention_grid_all_emotions.png` — one correctly-classified
  example per emotion, with attention overlaid
- `figures/18_top_attended_words_per_emotion.png` — corpus-level aggregate
  of the words each emotion relies on most

## Gradio Demo

Section 10 launches an interactive **Gradio** app (`share=True`) letting you:

- Type any sentence
- Pick a model (LSTM / GRU / BiLSTM+Attention / DistilBERT)
- See predicted emotion probabilities
- See the attention heatmap (for BiLSTM+Attention)

Run Section 10's cells in the notebook to launch it and get a shareable link.

## Limitations

- Single-label filtering discards a portion of the original multi-label
  GoEmotions corpus.
- Domain-specific to Reddit-style text; may not generalize to other domains
  without adaptation.
- Class imbalance persists despite weighted loss for rarer emotions.
- Attention weights show *where* the model looked, not a complete causal
  explanation of *why*.

## Future Work

- Extend to full multi-label (27-emotion) classification
- Train a self-attention/Transformer encoder from scratch for comparison
- Data augmentation for minority classes
- Model quantization/distillation for production deployment
- Calibration analysis (reliability diagrams)

## License

This project is released under the [MIT License](LICENSE). The GoEmotions
dataset is released by Google Research under its own license — see the
[dataset card](https://huggingface.co/datasets/google-research-datasets/go_emotions)
for details. Pretrained GloVe vectors are from
[Stanford NLP](https://nlp.stanford.edu/projects/glove/), and DistilBERT is
from [Hugging Face](https://huggingface.co/distilbert-base-uncased).
