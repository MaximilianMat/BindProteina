#!/usr/bin/env python

"""Convert a LoRA-tuned LaProteina checkpoint into an inference-ready checkpoint.

The project inference path currently loads checkpoints directly via
`Proteina.load_from_checkpoint(...)` without re-applying LoRA module replacement.
That means LoRA checkpoints (saved with `train.py` where modules are wrapped by
`loralib`) cannot be consumed directly by `generate.py` or `LaProteinaGenerator`.

This utility:
1) loads the source checkpoint
2) reconstructs the model configuration from `hyper_parameters`
3) re-wraps modules with LoRA using the same settings used in training
4) loads LoRA checkpoint weights
5) merges LoRA adapters into base weights
6) prunes LoRA-only tensors
7) writes a checkpoint with merged plain weights only
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Allow running this script directly from repository root.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from proteinfoundation.proteina import Proteina
from proteinfoundation.utils.lora_utils import replace_lora_layers


def _as_tensor(param_like, module_name: str, param_name: str) -> torch.Tensor:
    """Return a tensor from either lora parameter objects or parameter attributes."""
    if isinstance(param_like, torch.Tensor):
        return param_like
    if isinstance(param_like, nn.Parameter):
        return param_like
    # loralib can expose helper modules/objects with a `.weight` attribute.
    if hasattr(param_like, "weight") and isinstance(param_like.weight, (torch.Tensor, nn.Parameter)):
        return param_like.weight
    raise AttributeError(f"{module_name}: {param_name} is not a tensor-like object.")


def _cfg_get(cfg, key, default=None):
    """Read attribute or mapping key with a single shared helper."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            # OmegaConf DictConfig.get() can throw for unsupported default arg usage.
            return cfg.get(key) if key in cfg else default
    return getattr(cfg, key, default)


def _to_mapping(cfg):
    if isinstance(cfg, dict):
        return cfg
    try:
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return dict(cfg)


def _has_lora_in_checkpoint(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any("lora" in k.lower() for k in state_dict)


def _merge_candidate_lora_weights(A: torch.Tensor, B: torch.Tensor, weight_shape: torch.Size):
    """Try common LoRA weight composition conventions.

    Most LoRA wrappers use `B @ A`, but some projects use transposed layouts.
    We try both orientation patterns and return the first matching shape.
    """
    def _safe_matmul(lhs: torch.Tensor, rhs: torch.Tensor):
        try:
            return torch.matmul(lhs, rhs)
        except RuntimeError:
            return None

    candidates = []
    # Typical LoRA: B[out, r] @ A[r, in] => [out, in]
    candidates.append(("B@A", _safe_matmul(B, A)))
    # Alternate / transposed possibility.
    candidates.append(("A@B", _safe_matmul(A, B)))
    # Handle possible extra dimensions for vectorized module variants.
    candidates.append(("Bᵀ@Aᵀ", _safe_matmul(B.t(), A.t())))
    candidates.append(("Aᵀ@Bᵀ", _safe_matmul(A.t(), B.t())))
    candidates.append(("B@Aᵀ", _safe_matmul(B, A.t())))
    candidates.append(("Aᵀ@B", _safe_matmul(A.t(), B)))
    candidates.append(("Bᵀ@A", _safe_matmul(B.t(), A)))
    candidates.append(("A@Bᵀ", _safe_matmul(A, B.t())))

    candidate_shapes = []
    for label, candidate in candidates:
        if candidate is None:
            continue
        candidate_shapes.append((label, candidate.shape))
        if candidate.shape == weight_shape:
            return candidate

    # Last resort: attempt to reshape/transpose heuristically if one valid candidate exists.
    for _, candidate in candidates:
        if candidate is not None:
            flat = candidate.reshape(-1)
            if flat.numel() == int(torch.tensor(weight_shape).prod()):
                return flat.view(weight_shape)
            break

    raise RuntimeError(
        f"Could not align LoRA delta with weight shape {weight_shape}. "
        f"Got candidate shapes: {candidate_shapes}"
    )


def _merge_lora_module(module: nn.Module, name: str) -> Tuple[bool, str]:
    """Attempt to merge LoRA adapter parameters into the wrapped weight."""
    # Native API on lora.Linear / lora.Embedding
    merge_fn = getattr(module, "merge", None)
    if callable(merge_fn):
        merge_fn()
        return True, f"{name}: merged via merge()"

    merge_w_fn = getattr(module, "merge_weights", None)
    if callable(merge_w_fn):
        merge_w_fn()
        return True, f"{name}: merged via merge_weights()"

    # Fallback for manual weight merge.
    lora_A = getattr(module, "lora_A", None)
    lora_B = getattr(module, "lora_B", None)
    if lora_A is not None and lora_B is not None and hasattr(module, "weight"):
        scaling = float(getattr(module, "scaling", 1.0))
        with torch.no_grad():
            lora_A_data = _as_tensor(lora_A, name, "lora_A")
            lora_B_data = _as_tensor(lora_B, name, "lora_B")
            delta = _merge_candidate_lora_weights(
                lora_A_data, lora_B_data, module.weight.shape
            )
            module.weight.data.add_(delta * scaling)
        return True, f"{name}: merged via fallback lora_A/lora_B"

    return False, ""


def _collect_lora_module_names(model: nn.Module) -> Iterable[str]:
    # We treat parameters with "lora_" as LoRA-only state keys.
    return [name for name, _ in model.named_parameters() if "lora_" in name.lower()]


def merge_lora_checkpoint(
    input_ckpt: str,
    output_ckpt: str,
    lora_r: int = None,
    lora_alpha: float = None,
    lora_dropout: float = None,
    autoencoder_ckpt_path: str = None,
    keep_optimizer: bool = False,
    strict: bool = False,
    verify: bool = True,
) -> Dict:
    ckpt = torch.load(input_ckpt, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise ValueError(f"No state_dict found in checkpoint: {input_ckpt}")

    state_dict = ckpt["state_dict"]
    source_lora = _has_lora_in_checkpoint(state_dict)
    if not source_lora:
        msg = "No LoRA keys detected in source checkpoint state_dict."
        if strict:
            raise ValueError(msg)
        print(f"[WARN] {msg} Proceeding (this may be already-merged).")

    # Build model config from checkpoint hyper-parameters (created by Proteina.save_hyperparameters()).
    hp = ckpt.get("hyper_parameters", {})
    cfg_exp = (
        hp.get("cfg_exp")
        or hp.get("cfg")
        or hp.get("lightning_module")
        or hp.get("model_config")
    )
    if cfg_exp is None:
        raise ValueError(
            "Cannot determine model config from checkpoint hyper_parameters. "
            "Expected `cfg_exp` but it was not found."
        )
    cfg_exp = OmegaConf.create(_to_mapping(cfg_exp))

    # Resolve LoRA hyperparameters from checkpoint first, then CLI overrides.
    cfg_lora = _cfg_get(cfg_exp, "lora", {})
    resolved_r = _cfg_get(cfg_lora, "r", None)
    resolved_alpha = _cfg_get(cfg_lora, "lora_alpha", None)
    resolved_dropout = _cfg_get(cfg_lora, "lora_dropout", None)
    if lora_r is not None:
        resolved_r = lora_r
    if lora_alpha is not None:
        resolved_alpha = lora_alpha
    if lora_dropout is not None:
        resolved_dropout = lora_dropout

    if resolved_r is None:
        raise ValueError(
            "Could not resolve LoRA rank. Provide --lora_r or ensure checkpoint includes cfg_exp.lora.r."
        )
    if resolved_alpha is None:
        raise ValueError(
            "Could not resolve LoRA alpha. Provide --lora_alpha or ensure checkpoint includes cfg_exp.lora.lora_alpha."
        )
    if resolved_dropout is None:
        raise ValueError(
            "Could not resolve LoRA dropout. Provide --lora_dropout or ensure checkpoint includes cfg_exp.lora.lora_dropout."
        )
    try:
        resolved_r = int(resolved_r)
        resolved_alpha = float(resolved_alpha)
        resolved_dropout = float(resolved_dropout)
    except Exception as err:
        raise ValueError("LoRA hyperparameters must be numeric (r, alpha, dropout).") from err
    if resolved_r <= 0:
        raise ValueError(f"Invalid LoRA rank r={resolved_r}; expected positive integer.")
    if resolved_alpha <= 0:
        raise ValueError(f"Invalid LoRA alpha={resolved_alpha}; expected positive value.")
    if not (0.0 <= resolved_dropout < 1.0):
        raise ValueError(f"Invalid LoRA dropout={resolved_dropout}; expected 0 <= dropout < 1.")

    # Optional path override for autoencoder.
    if autoencoder_ckpt_path is not None:
        cfg_exp = OmegaConf.to_container(cfg_exp, resolve=True)
        cfg_exp["autoencoder_ckpt_path"] = autoencoder_ckpt_path
        cfg_exp = OmegaConf.create(cfg_exp)

    # Build model and wrap LoRA modules in the same way training does.
    model = Proteina(cfg_exp)
    replace_lora_layers(model, resolved_r, resolved_alpha, resolved_dropout)

    # Load LoRA checkpoint weights into the wrapped model.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            "Strict mode requested but checkpoint loading had mismatches.\n"
            f"missing={missing}\nunexpected={unexpected}"
        )
    if missing:
        print(
            f"[WARN] Missing keys when loading source LoRA checkpoint: "
            f"{len(missing)} (showing up to 10): {list(missing)[:10]}"
        )
    if unexpected:
        print(
            f"[WARN] Unexpected keys when loading source LoRA checkpoint: "
            f"{len(unexpected)} (showing up to 10): {list(unexpected)[:10]}"
        )

    # Merge adapters.
    merged = 0
    merge_debug = []
    lora_module_names = _collect_lora_module_names(model)
    for module_name, module in model.named_modules():
        did_merge, detail = _merge_lora_module(module, module_name)
        if did_merge:
            merged += 1
            merge_debug.append(detail)
    if merged == 0 and lora_module_names and source_lora:
        raise RuntimeError(
            "No LoRA modules were merged. Model appears to be missing LoRA wrappers. "
            "Inspect checkpoint/model structure."
        )

    # Export merged plain weights.
    merged_state = {
        k: v for k, v in model.state_dict().items() if "lora_" not in k.lower()
    }
    if source_lora and not _has_lora_in_checkpoint(merged_state):
        print(f"[INFO] Removed LoRA-only keys; resulting state_dict has {len(merged_state)} tensors.")

    output = dict(ckpt)
    output["state_dict"] = merged_state
    if not keep_optimizer:
        for key in ["optimizer_states", "lr_schedulers", "callback_states"]:
            output.pop(key, None)

    output_dir = Path(output_ckpt).expanduser().resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(output, output_ckpt)
    print(
        f"[OK] Wrote merged checkpoint to: {output_ckpt}\n"
        f"LoRA merge attempts: {merged}; sample details: {merge_debug[:3]}"
    )

    if verify:
        # Round-trip load to guarantee existing inference path can at least parse this file.
        reloaded = torch.load(output_ckpt, map_location="cpu", weights_only=False)
        if "lora_" in "".join(reloaded["state_dict"].keys()).lower():
            raise RuntimeError("Converted checkpoint still contains LoRA keys after pruning.")

        _ = Proteina.load_from_checkpoint(
            output_ckpt,
            strict=False,
            autoencoder_ckpt_path=autoencoder_ckpt_path,
        )
        print("[OK] Validation load succeeded with Proteina.load_from_checkpoint(..., strict=False).")

    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge LoRA adapter weights into base parameters and write a checkpoint "
            "that existing inference scripts can load directly."
        )
    )
    parser.add_argument("--input_ckpt", required=True, help="Path to source LoRA checkpoint (.ckpt)")
    parser.add_argument("--output_ckpt", required=True, help="Destination path for merged checkpoint")
    parser.add_argument("--lora_r", type=int, default=None, help="Override LoRA rank r")
    parser.add_argument("--lora_alpha", type=float, default=None, help="Override LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=None, help="Override LoRA dropout")
    parser.add_argument(
        "--autoencoder_ckpt_path",
        type=str,
        default=None,
        help="Optional AE checkpoint override used for verification load",
    )
    parser.add_argument(
        "--keep-optimizer",
        action="store_true",
        default=False,
        help="Keep optimizer/lr scheduler/callback states in output checkpoint",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail when source checkpoint appears to have no LoRA keys or load mismatch is non-empty",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        default=False,
        help="Skip post-conversion Proteina load verification",
    )
    return parser.parse_args()


def _maybe_copy_or_warn(src: str, dst: str):
    """Handle overwrite semantics safely."""
    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError("input_ckpt and output_ckpt are the same file.")
    if os.path.exists(dst):
        raise FileExistsError(f"output_ckpt already exists: {dst}. Move/remove it first.")


def main():
    args = parse_args()
    _maybe_copy_or_warn(args.input_ckpt, args.output_ckpt)
    merge_lora_checkpoint(
        input_ckpt=args.input_ckpt,
        output_ckpt=args.output_ckpt,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        autoencoder_ckpt_path=args.autoencoder_ckpt_path,
        keep_optimizer=args.keep_optimizer,
        strict=args.strict,
        verify=not args.skip_verify,
    )


if __name__ == "__main__":
    main()
