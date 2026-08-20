"""
Load published Qwen3-0.6B weights into the from-scratch model and verify them.

Weights are cached in `.hf_cache/` at the repo root, which is gitignored and
safe to delete once you are done.

Usage:
    python src/architecture/load_qwen.py            # download, remap, verify, generate
    python src/architecture/load_qwen.py --no-ref   # skip the HuggingFace comparison
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Config, KVCache, LanguageModel  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / ".hf_cache"
MODEL_ID = "Qwen/Qwen3-0.6B"


def remap(sd: dict[str, torch.Tensor], cfg: Config) -> dict[str, torch.Tensor]:
    """
    Map published parameter names onto our module names.

    The tensors themselves are used as-is: nn.Linear stores its weight as
    [out_features, in_features], which is exactly how the checkpoint stores it.
    Transposing here is the classic way to "fix" a shape that was never wrong.
    """
    out = {"embed.weight": sd["model.embed_tokens.weight"], "norm.weight": sd["model.norm.weight"]}

    for i in range(cfg.n_layers):
        src, dst = f"model.layers.{i}.", f"blocks.{i}."
        out[dst + "input_norm.weight"] = sd[src + "input_layernorm.weight"]
        out[dst + "post_attn_norm.weight"] = sd[src + "post_attention_layernorm.weight"]
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            out[f"{dst}attn.{p}.weight"] = sd[f"{src}self_attn.{p}.weight"]
        for p in ("q_norm", "k_norm"):
            out[f"{dst}attn.{p}.weight"] = sd[f"{src}self_attn.{p}.weight"]
        for p in ("gate_proj", "up_proj", "down_proj"):
            out[f"{dst}ffn.{p}.weight"] = sd[f"{src}mlp.{p}.weight"]

    # tie_word_embeddings: true, so the checkpoint carries no lm_head.weight.
    # Leaving it randomly initialized produces fluent internals and pure garbage out.
    out["head.weight"] = out["embed.weight"]
    return out


@torch.no_grad()
def generate(model, ids, max_new_tokens=40, temperature=0.7, top_p=0.9, eos_id=None):
    """Prefill the prompt, then decode one token at a time through the KV cache."""
    caches = [KVCache() for _ in model.blocks]
    logits = model(ids, caches)[:, -1]                       # PREFILL: whole prompt at once

    for _ in range(max_new_tokens):
        if temperature == 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            srt, idx = probs.sort(descending=True, dim=-1)
            keep = (srt.cumsum(-1) - srt) < top_p            # nucleus
            srt = torch.where(keep, srt, torch.zeros_like(srt))
            nxt = idx.gather(-1, torch.multinomial(srt / srt.sum(-1, keepdim=True), 1))

        ids = torch.cat([ids, nxt], dim=1)
        if eos_id is not None and nxt.item() == eos_id:
            break
        logits = model(nxt, caches)[:, -1]                   # DECODE: one token per step
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ref", action="store_true", help="skip HuggingFace logit comparison")
    ap.add_argument("--prompt", default="The CEO announced record earnings on Friday")
    args = ap.parse_args()

    CACHE_DIR.mkdir(exist_ok=True)
    os.environ["HF_HOME"] = str(CACHE_DIR)                   # keep every artifact in-repo
    print(f"cache dir: {CACHE_DIR}  (gitignored, safe to delete)\n")

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    path = Path(snapshot_download(MODEL_ID, cache_dir=CACHE_DIR))
    print(f"weights: {path}\n")

    cfg = Config()
    sd = load_file(path / "model.safetensors")
    print(f"checkpoint tensors: {len(sd)}")
    for name in list(sd)[:3]:
        print(f"  {name:52s} {tuple(sd[name].shape)}")

    # ---- build ours and load ----
    model = LanguageModel(cfg)
    missing, unexpected = model.load_state_dict(remap(sd, cfg), strict=True), None
    model = model.to(torch.float32).eval()

    counts = model.n_params()
    real = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    print(f"\nparameters: {counts['total']:,} predicted | {real:,} actual")
    for k in ("attention", "ffn", "norms", "embedding"):
        print(f"  {k:10s} {counts[k]:>12,}  {100 * counts[k] / counts['total']:5.2f}%")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    ids = tok(args.prompt, return_tensors="pt").input_ids

    # ---- verify against the reference implementation ----
    if not args.no_ref:
        from transformers import AutoModelForCausalLM

        ref = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, cache_dir=CACHE_DIR, dtype=torch.float32
        ).eval()
        with torch.no_grad():
            ours, theirs = model(ids), ref(ids).logits
        diff = (ours - theirs).abs().max().item()
        same = torch.equal(ours.argmax(-1), theirs.argmax(-1))
        print(f"\nmax |logit diff| : {diff:.3e}")
        print(f"argmax identical : {same}")
        assert same and diff < 1e-2, "implementation does not match the reference"
        print("VERIFIED: from-scratch implementation matches the reference")
        del ref

    # ---- generate ----
    print(f"\nprompt: {args.prompt!r}")
    out = generate(model, ids, max_new_tokens=40, temperature=0.0, eos_id=tok.eos_token_id)
    print(f"greedy: {tok.decode(out[0], skip_special_tokens=True)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
