#!/usr/bin/env python3
"""Probe the ported logprob_token_ids fill on the real AMD Triton backend.

Run by the orchestrator INSIDE the serve container:

    python3 probe_logprob_token_ids_kernel.py /path/to/vllm_sample_logprob.py \
        [--device 0]

Standalone: imports the derived module by file path, builds tiny tensors on
cuda:<index> (batch 3, vocab 64), runs the fill through the triton kernel and the
pure-torch twin, compares both against a pure-Python reference, prints one JSON line

    {"triton": "ok"|"<error>", "torch": "ok"|"<error>", "equal": bool,
     "triton_version": ...}

and exits 0 only if the torch path is ok. Allocates well under 64 MB and never
imports vllm's engine. If `vllm.triton_utils` (or the module itself) is not
importable, that is reported clearly in the JSON line and reflected in the exit
code (non-zero).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys


def _reference(sampled, topk, num_topk, expanded, num_per, per, num_cols):
    """Pure-Python ground truth, host ints only."""
    out, valid = [], []
    for b in range(len(sampled)):
        row = [int(sampled[b])]
        v = [True]
        r = int(expanded[b])
        n = int(num_per[r])
        if n > 0:
            row += [int(per[r][i]) for i in range(n)]
            v += [True] * n
        else:
            row += [int(topk[b][i]) for i in range(num_topk)]
            v += [True] * num_topk
        row += [0] * (num_cols + 1 - len(row))
        v += [False] * (num_cols + 1 - len(v))
        out.append(row)
        valid.append(v)
    return out, valid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("module", help="path to the derived vllm_sample_logprob.py")
    ap.add_argument("--device", type=int, default=0, help="cuda device index")
    args = ap.parse_args(argv)

    out = {"triton": "error", "torch": "error", "equal": False,
           "triton_version": None}

    try:
        import triton  # noqa: F401
        out["triton_version"] = triton.__version__
    except Exception as exc:
        out["triton"] = f"triton import failed: {type(exc).__name__}: {exc}"
        print(json.dumps(out))
        return 1

    try:
        import torch
    except Exception as exc:
        out["torch"] = f"torch import failed: {type(exc).__name__}: {exc}"
        print(json.dumps(out))
        return 1

    spec = importlib.util.spec_from_file_location(
        "_probed_vllm_sample_logprob", args.module)
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:
        msg = f"module import failed: {type(exc).__name__}: {exc}"
        out["triton"] = msg
        out["torch"] = msg
        print(json.dumps(out))
        return 1

    try:
        from vllm.triton_utils import triton  # the backend the serve uses
        out["triton_version"] = triton.__version__
    except Exception as exc:
        out["triton"] = f"vllm.triton_utils import failed: {type(exc).__name__}: {exc}"
        print(json.dumps(out))
        return 1

    device = f"cuda:{args.device}"
    B, V = 3, 64
    num_topk = 4
    num_reqs = 3
    max_per = 6

    sampled = torch.randint(0, V, (B,), dtype=torch.int64, device=device)
    logits = torch.randn(B, V, device=device)
    topk = torch.topk(logits, num_topk, dim=-1).indices.to(torch.int32)
    expanded = torch.arange(num_reqs, dtype=torch.int32, device=device)
    # one custom row (n=2 < num_topk), one pure-topk row (n=0), one n == max_per row
    num_per = torch.tensor([2, 0, max_per], dtype=torch.int32, device=device)
    per = torch.zeros(num_reqs, max_per, dtype=torch.int32, device=device)
    per[0, :2] = torch.tensor([7, 9], dtype=torch.int32, device=device)
    per[2, :max_per] = torch.arange(10, 10 + max_per, dtype=torch.int32,
                                    device=device)
    num_cols = max(num_topk, max_per)

    ref_ids, ref_valid = _reference(
        sampled.tolist(), topk.tolist(), num_topk, expanded.tolist(),
        num_per.tolist(), per.tolist(), num_cols)

    # --- triton path (mirrors compute_topk_scores: uint8 mask, .bool() after) ---
    try:
        ids_k = sampled.new_zeros((B, 1 + num_cols))
        mask_k = torch.zeros_like(ids_k, dtype=torch.uint8)
        mod._fill_logprob_token_ids_kernel[(B,)](
            ids_k, ids_k.stride(0), mask_k, mask_k.stride(0),
            sampled, topk, topk.stride(0), expanded,
            num_per, per, per.stride(0),
            NUM_TOPK=num_topk,
            PADDED_COLS=triton.next_power_of_2(num_cols),
        )
        torch.cuda.synchronize(device)
        got_k = ids_k.tolist()
        got_kv = mask_k.bool().tolist()
        ok_k = (got_k == ref_ids and got_kv == ref_valid)
        out["triton"] = "ok" if ok_k else f"MISMATCH: ids={got_k} valid={got_kv}"
    except Exception as exc:
        out["triton"] = f"{type(exc).__name__}: {exc}"

    # --- torch path ---
    try:
        ids_t, valid_t = mod._fill_logprob_token_ids_torch(
            sampled, topk, num_topk, expanded, num_per, per, num_cols)
        got_t = ids_t.tolist()
        got_tv = valid_t.tolist()
        ok_t = (got_t == ref_ids and got_tv == ref_valid)
        out["torch"] = "ok" if ok_t else f"MISMATCH: ids={got_t} valid={got_tv}"
    except Exception as exc:
        out["torch"] = f"{type(exc).__name__}: {exc}"

    out["equal"] = out["triton"] == "ok" and out["torch"] == "ok"
    print(json.dumps(out))
    return 0 if out["torch"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
