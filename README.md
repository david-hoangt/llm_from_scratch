# NLP From Scratch

Every component of a modern large language model, implemented from zero in PyTorch and
explained in a companion blog post. Attention and its efficiency variants, rotary positional
encoding, normalization, activations, byte-pair-encoding tokenizers, and finally a complete
decoder-only transformer that **loads the published Qwen3-0.6B weights and reproduces the
reference implementation exactly** (max logit difference `0.000e+00`).

Written for people who want to understand LLM internals by building them, not by reading about
them. Every notebook runs end to end, on a free Colab GPU or on CPU.

**Blog:** [backpropolis.github.io](https://backpropolis.github.io)

## Notebooks

| # | Notebook | What it builds | Colab | Blog post |
|---|----------|----------------|-------|-----------|
| 0 | [0. neural_network_from_scratch.ipynb](https://github.com/backpropolis/nlp_from_scratch/blob/main/src/0.neural_net/0.%20neural_network_from_scratch.ipynb) | Neural network, backprop, SGD | [Open](https://colab.research.google.com/github/backpropolis/nlp_from_scratch/blob/main/src/0.neural_net/0.%20neural_network_from_scratch.ipynb) | [Neural Network From Scratch](https://backpropolis.github.io/posts/2026-03-23-neural-net/) |
| 1 | [1. self_attention.ipynb](https://github.com/backpropolis/nlp_from_scratch/blob/main/src/attention/1.%20self_attention.ipynb) | Self-attention, Q/K/V, causal masking | [Open](https://colab.research.google.com/github/backpropolis/nlp_from_scratch/blob/main/src/attention/1.%20self_attention.ipynb) | [Self-Attention and Multi-Head Attention](https://backpropolis.github.io/posts/2026-03-30-self-attn/) |
| 2 | [2. mutihead_attention.ipynb](https://github.com/backpropolis/nlp_from_scratch/blob/main/src/attention/2.%20mutihead_attention.ipynb) | Multi-head attention | [Open](https://colab.research.google.com/github/backpropolis/nlp_from_scratch/blob/main/src/attention/2.%20mutihead_attention.ipynb) | [Self-Attention and Multi-Head Attention](https://backpropolis.github.io/posts/2026-03-30-self-attn/) |
| 3 | [3. cross_attention.ipynb](https://github.com/backpropolis/nlp_from_scratch/blob/main/src/attention/3.%20cross_attention.ipynb) | Cross-attention, padding masks | [Open](https://colab.research.google.com/github/backpropolis/nlp_from_scratch/blob/main/src/attention/3.%20cross_attention.ipynb) | [Self-Attention and Multi-Head Attention](https://backpropolis.github.io/posts/2026-03-30-self-attn/) |
| 4 | [4. attention_variants.ipynb](https://github.com/backpropolis/nlp_from_scratch/blob/main/src/attention/4.%20attention_variants.ipynb) | GQA, MLA, sliding window, hybrid | [Open](https://colab.research.google.com/github/backpropolis/nlp_from_scratch/blob/main/src/attention/4.%20attention_variants.ipynb) | [Attention Variants](https://backpropolis.github.io/posts/2026-04-09-attn-variant/) |
| 5 | [5. positional_encoding.ipynb](https://github.com/backpropolis/nlp_from_scratch/blob/main/src/positional_encoding/5.%20positional_encoding.ipynb) | Sinusoidal, learned, ALiBi, RoPE, NoPE | [Open](https://colab.research.google.com/github/backpropolis/nlp_from_scratch/blob/main/src/positional_encoding/5.%20positional_encoding.ipynb) | [Positional Encoding](https://backpropolis.github.io/posts/2026-05-06-positional-encoding/) |
| 6 | [6. normalization.ipynb](https://github.com/backpropolis/nlp_from_scratch/blob/main/src/normalization/6.%20normalization.ipynb) | LayerNorm, RMSNorm, QK-Norm, DyT | [Open](https://colab.research.google.com/github/backpropolis/nlp_from_scratch/blob/main/src/normalization/6.%20normalization.ipynb) | [Normalization](https://backpropolis.github.io/posts/2026-06-09-normalization/) |
| 7 | [7. activations.ipynb](https://github.com/backpropolis/nlp_from_scratch/blob/main/src/activations/7.%20activations.ipynb) | ReLU, GELU, SiLU, SwiGLU, ReLU² | [Open](https://colab.research.google.com/github/backpropolis/nlp_from_scratch/blob/main/src/activations/7.%20activations.ipynb) | [Activations](https://backpropolis.github.io/posts/2026-06-29-activations/) |
| 8 | [8. tokenizers.ipynb](https://github.com/backpropolis/nlp_from_scratch/blob/main/src/tokenizers/8.%20tokenizers.ipynb) | BPE from scratch, byte-level, WordPiece | [Open](https://colab.research.google.com/github/backpropolis/nlp_from_scratch/blob/main/src/tokenizers/8.%20tokenizers.ipynb) | [Tokenizers](https://backpropolis.github.io/posts/2026-07-05-tokenizer/) |
| 9 | [9. architecture.ipynb](https://github.com/backpropolis/nlp_from_scratch/blob/main/src/architecture/9.%20architecture.ipynb) | Full dense LLM, loads real Qwen3-0.6B weights | [Open](https://colab.research.google.com/github/backpropolis/nlp_from_scratch/blob/main/src/architecture/9.%20architecture.ipynb) | [Architecture](https://backpropolis.github.io/posts/2026-08-20-architecture/) |

## The architecture notebook

Notebook 9 is the one that ties the series together. It assembles RMSNorm, RoPE, grouped-query
attention with QK-Norm, and a SwiGLU feed-forward network into a 596M-parameter decoder-only
model, then loads the real Qwen3-0.6B checkpoint into that from-scratch implementation:

```
parameters: 596,049,920
max |logit diff| : 0.000e+00
argmax identical : True
VERIFIED: every component matches the reference implementation
```

Coherent text from weights you did not train is a much stronger check than a training curve.
It only happens when every convention lines up: the split-half RoPE layout, the grouped-query
head expansion, QK-Norm before the rotation, and the tied LM head.

## Setup

```bash
# Python 3.13+
uv sync
```

Or open any notebook in Colab with the links above, which installs what it needs.

## Topics covered

Transformer internals, self-attention, multi-head attention, cross-attention, grouped-query
attention (GQA), multi-latent attention (MLA), sliding-window attention, RoPE, ALiBi, NoPE,
LayerNorm, RMSNorm, QK-Norm, Dynamic Tanh, ReLU, GELU, SiLU, SwiGLU, gated FFNs, byte-pair
encoding, byte-level BPE, WordPiece, Unigram/SentencePiece, KV caching, prefill and decode,
weight tying, and loading published checkpoints.

## Stack

PyTorch · NumPy · Matplotlib · Jupyter · managed with `uv`
