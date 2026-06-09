# HMM Part-of-Speech Tagger

A Hidden Markov Model (HMM) POS tagger built **from scratch** with log-space Viterbi decoding, trained on the Brown Corpus (universal tagset).

## Features

- **HMM from scratch** — transition & emission probabilities estimated via counting, Laplace smoothing for OOV words
- **Log-space Viterbi** — avoids floating-point underflow on long sentences
- **Unknown word handling** — suffix backoff + capitalization clues learned from rare words
- **Flask web app** — type a sentence and see POS tags in real-time
- **Jupyter notebook** — step-by-step training, evaluation, and visualizations

## Quick Start

```bash
pip install flask nltk
python app.py
# Open http://127.0.0.1:5000
```

## Files

| File | Description |
|------|-------------|
| `app.py` | Flask web app with full HMM implementation |
| `hmm_pos_tagger.ipynb` | Jupyter notebook with training & evaluation |
| `prepare_enhancements.py` | Generates accuracy plots and visualizations |
| `figures/` | Generated plots (accuracy, tag distribution, transition heatmap) |

## Tag Set (Universal)

`NOUN`, `VERB`, `ADJ`, `ADV`, `PRON`, `DET`, `ADP`, `CONJ`, `NUM`, `PRT`, `X`, `.`

## Evaluation

Tested on 20% holdout of the Brown Corpus. The model achieves competitive accuracy using pure HMM + smoothing, without any neural networks.
