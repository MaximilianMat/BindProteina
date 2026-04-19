#!/usr/bin/env python
import argparse, os, collections, torch

def summarize_ckpt(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", {})
    keys = list(state.keys())
    lora = [k for k in keys if "lora_" in k.lower() or ".lora_" in k.lower()]

    print(f"\n== {path} ==")
    print(f"total state_dict keys: {len(keys)}")
    print(f"lora-like keys: {len(lora)}")

    if not lora:
        return

    scope = collections.Counter()
    module3 = collections.Counter()
    parent = collections.Counter()
    contains = collections.Counter()

    for k in lora:
        parts = k.split(".")
        if parts:
            scope[parts[0]] += 1
            module3[".".join(parts[:3])] += 1
            lk = k.lower()
            for tok in [
                "autoencoder",
                "encoder",
                "decoder",
                "transformer",
                "model",
                "proteina",
                "lm",
                "ffn",
                "attention",
            ]:
                if tok in lk:
                    contains[tok] += 1

    print("LoRA by first segment:")
    for k, v in scope.most_common():
        print(f"  {k}: {v}")
    print("LoRA by first 3 segments:")
    for k, v in module3.most_common(30):
        print(f"  {k}: {v}")
    print("Keyword hits:")
    for k, v in contains.most_common():
        print(f"  {k}: {v}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    args = ap.parse_args()
    for p in args.ckpts:
        if os.path.exists(p):
            summarize_ckpt(p)
        else:
            print(f"[MISSING] {p}")

if __name__ == "__main__":
    main()