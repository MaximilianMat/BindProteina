#!/usr/bin/env python
import argparse, torch
from pathlib import Path
from omegaconf import OmegaConf
from laproteina.proteinfoundation.proteina import Proteina

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repo root if script is in scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proteinfoundation.utils.lora_utils import replace_lora_layers

def _to_mapping(cfg):
    if isinstance(cfg, dict):
        return cfg
    try:
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return dict(cfg)

def load_cfg_exp(ckpt):
    hp = ckpt.get("hyper_parameters", {})
    cfg_exp = hp.get("cfg_exp") or hp.get("cfg") or hp.get("lightning_module") or hp.get("model_config")
    if cfg_exp is None:
        raise ValueError("cfg_exp not found in checkpoint hyper_parameters")
    return OmegaConf.create(_to_mapping(cfg_exp))

def resolve_lora(cfg_exp):
    lora = cfg_exp.get("lora", {})
    return int(lora["r"]), float(lora["lora_alpha"]), float(lora["lora_dropout"])

def get_merged_state_from_source(source_ckpt):
    ckpt = torch.load(source_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    cfg_exp = load_cfg_exp(ckpt)
    r, a, d = resolve_lora(cfg_exp)

    model = Proteina(cfg_exp)
    replace_lora_layers(model, r, a, d)
    model.load_state_dict(state, strict=False)

    # merge in-memory
    from proteinfoundation.merge_lora_checkpoint import _merge_lora_module
    for name, mod in model.named_modules():
        _merge_lora_module(mod, name)

    return model.state_dict()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--merged", required=True)
    args = ap.parse_args()

    src_state = get_merged_state_from_source(args.source)
    merged_ckpt = torch.load(args.merged, map_location="cpu", weights_only=False)["state_dict"]

    # compare all non-LoRA keys
    diffs = []
    for k, v in src_state.items():
        if "lora_" in k.lower():
            continue
        if k not in merged_ckpt:
            print("MISSING IN MERGED:", k)
            continue
        a = v.detach().float().cpu()
        b = merged_ckpt[k].detach().float().cpu()
        if a.shape != b.shape:
            print("SHAPE MISMATCH:", k, a.shape, b.shape)
            continue
        max_abs = (a - b).abs().max().item()
        mean_abs = (a - b).abs().mean().item()
        if max_abs > 1e-5:
            diffs.append((k, max_abs, mean_abs))

    print(f"Compared keys: {len(src_state)}")
    print(f"max nonzero-diff keys: {len(diffs)}")
    for k, mx, me in sorted(diffs, key=lambda t: t[1], reverse=True)[:30]:
        print(f"{k} | max_abs={mx:.3e} | mean_abs={me:.3e}")

if __name__ == "__main__":
    main()