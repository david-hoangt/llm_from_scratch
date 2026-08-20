# LLM From Scratch

Building LLM internals from scratch in PyTorch — attention mechanisms, one notebook at a time.

## Attention

| Notebook | Mechanism | What it covers |
|----------|-----------|---------------|
| [1. self_attention.ipynb](attention/1.%20self_attention.ipynb) | Self-Attention | Q/K/V projections, scaled dot-product, bidirectional & causal modes, attention heatmaps |
| [2. mutihead_attention.ipynb](attention/2.%20mutihead_attention.ipynb) | Multi-Head Attention | h parallel heads in d_model/h subspaces, W_O output projection, per-head visualization |
| [3. cross_attention.ipynb](attention/3.%20cross_attention.ipynb) | Cross-Attention | Rectangular attention matrix, encoder-decoder wiring, padding masks, translation alignment |

## Blog

Companion blog post with diagrams and architectural context:

- [The Mechanics of Attention: Self, Multi-Head, Causal, and Cross](https://davidwhiteboard.github.io) — covers all four attention variants with the running example *"The CEO announced record earnings on Friday"*

## Setup

```bash
# Requires Python 3.13+
uv sync
```

## Dependencies

- PyTorch
- NumPy
- Matplotlib
- Seaborn
