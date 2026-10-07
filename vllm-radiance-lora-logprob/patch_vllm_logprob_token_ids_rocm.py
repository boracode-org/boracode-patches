#!/usr/bin/env python3
"""Make vLLM's `logprob_token_ids` fill kernel survivable on the AMD Triton backend.

TARGET (mounted over the container's vllm_sample_logprob.py):
    vllm_sample_logprob.py
    Preimage  sha256 78de8c1e2f338e0c83663217ac50950557aa77c6b52b5b76bd53876c51b89566
    Postimage sha256 c5f43d20eda1ce0c5dc2e4ccf9d7d8f6dca5ae2341d0ca530ffdc8765e27ae16

Usage: patch_vllm_logprob_token_ids_rocm.py --check <path> | --apply <path> | --print-hashes

WHY (measured 2026-10-05 on the bench: vLLM 0.27.1, 4x Radeon R9700 gfx1201, AMD Triton
backend): one /v1/completions request carrying `logprob_token_ids` killed the fleet
serve -- `compute_topk_scores` -> `_fill_logprob_token_ids_kernel[(batch_size,)]` ->
`triton/backends/amd/compiler.py make_ttgir` -> `RuntimeError: PassManager::run failed`,
raised inside the worker and FATAL to EngineCore. Every other kernel in this module
(`_topk_log_softmax_kernel`, `_ranks_kernel`) compiles fine on the same backend.

Suspects fixed here, the two things only this kernel did:

1. A runtime `if num_custom > 0:` selecting between two POINTER bases. Replaced with
   two independent masked loads combined by `tl.where` -- no runtime branch over
   pointers.
2. `tl.store(..., tl.full([PADDED_COLS], 1, tl.int1), ...)` into a bool tensor.
   Replaced with integer 0/1 stores into a `torch.uint8` buffer that the caller
   converts with `.bool()` after the call.

Plus a safety net for whatever remains broken: `compute_topk_scores` guards the kernel
call; on the FIRST exception it logs one warning, records the failure in a module-level
flag, and uses the pure-torch twin `_fill_logprob_token_ids_torch` from then on (never
retries the kernel, never raises). Env `HORDE_LOGPROB_TOKEN_IDS_IMPL` = `auto`
(default) | `torch` | `triton`: `torch` skips the kernel entirely, `triton` disables
the fallback (for probing). Nothing else in the module changes.
"""

from __future__ import annotations

import argparse
import hashlib
import sys

PREIMAGE_SHA256 = "78de8c1e2f338e0c83663217ac50950557aa77c6b52b5b76bd53876c51b89566"
POSTIMAGE_SHA256 = "c5f43d20eda1ce0c5dc2e4ccf9d7d8f6dca5ae2341d0ca530ffdc8765e27ae16"

REPLACEMENTS: list[tuple[str, str]] = [
    (
        "import numpy as np\nimport torch\n",
        "import logging\nimport os\n\nimport numpy as np\nimport torch\n",
    ),
    (
        "# Upper bound on the topk kernel's per-iteration gather width.\n"
        "_MAX_TOPK_BLOCK = 1024\n",
        "# Upper bound on the topk kernel's per-iteration gather width.\n"
        "_MAX_TOPK_BLOCK = 1024\n\n"
        "# HORDE_LOGPROB_TOKEN_IDS_IMPL: 'auto' (default) tries the triton kernel and\n"
        "# permanently falls back to _fill_logprob_token_ids_torch on its first\n"
        "# exception; 'torch' never runs the kernel; 'triton' never falls back\n"
        "# (probing). Read once at import.\n"
        "_LOGPROB_TOKEN_IDS_IMPL = os.environ.get(\"HORDE_LOGPROB_TOKEN_IDS_IMPL\", \"auto\")\n"
        "# First triton fill failure seen in this process (list-as-flag: mutation\n"
        "# needs no `global`).\n"
        "_TRITON_FILL_FAILED: list[bool] = []\n",
    ),
    (
        '@triton.jit\ndef _fill_logprob_token_ids_kernel(\n'
        '    # [batch_size, 1 + num_cols]\n'
        '    out_token_ids_ptr,\n'
        '    out_token_ids_stride,\n'
        '    # [batch_size, 1 + num_cols]\n'
        '    out_valid_mask_ptr,\n'
        '    out_valid_mask_stride,\n'
        '    sampled_token_ids_ptr,  # [batch_size]\n'
        '    topk_indices_ptr,  # [batch_size, NUM_TOPK] (unused when NUM_TOPK == 0)\n'
        '    topk_indices_stride,\n'
        '    expanded_idx_mapping_ptr,  # [batch_size] -> req_state_idx\n'
        '    num_per_req_token_ids_ptr,  # [max_num_reqs]\n'
        '    per_req_token_ids_ptr,  # [max_num_reqs, MAX_LOGPROB_TOKEN_IDS]\n'
        '    per_req_token_ids_stride,\n'
        '    NUM_TOPK: tl.constexpr,\n'
        '    PADDED_COLS: tl.constexpr,\n'
        '):\n'
        '    batch_idx = tl.program_id(0)\n'
        '\n'
        '    # Column 0: always the sampled token, always valid.\n'
        '    sampled = tl.load(sampled_token_ids_ptr + batch_idx)\n'
        '    tl.store(out_token_ids_ptr + batch_idx * out_token_ids_stride, sampled)\n'
        '    tl.store(out_valid_mask_ptr + batch_idx * out_valid_mask_stride, 1)\n'
        '\n'
        '    req_state_idx = tl.load(expanded_idx_mapping_ptr + batch_idx)\n'
        '    num_custom = tl.load(num_per_req_token_ids_ptr + req_state_idx)\n'
        '\n'
        '    col = tl.arange(0, PADDED_COLS)\n'
        '    tid_base = out_token_ids_ptr + batch_idx * out_token_ids_stride + 1\n'
        '    mask_base = out_valid_mask_ptr + batch_idx * out_valid_mask_stride + 1\n'
        '\n'
        '    if num_custom > 0:\n'
        '        # Override topk with per-request custom tokens.\n'
        '        src = per_req_token_ids_ptr + req_state_idx * per_req_token_ids_stride\n'
        '        valid = col < num_custom\n'
        '    else:\n'
        '        # Fill with topk indices (no-op when NUM_TOPK == 0).\n'
        '        src = topk_indices_ptr + batch_idx * topk_indices_stride\n'
        '        valid = col < NUM_TOPK\n'
        '\n'
        '    tokens = tl.load(src + col, mask=valid, other=0).to(tl.int64)\n'
        '    tl.store(tid_base + col, tokens, mask=valid)\n'
        '    tl.store(mask_base + col, tl.full([PADDED_COLS], 1, tl.int1), mask=valid)\n',
        # ------------------------------------------------------------------
        '@triton.jit\ndef _fill_logprob_token_ids_kernel(\n'
        '    # [batch_size, 1 + num_cols]\n'
        '    out_token_ids_ptr,\n'
        '    out_token_ids_stride,\n'
        '    # [batch_size, 1 + num_cols] uint8; the caller allocates torch.uint8 and\n'
        '    # converts with .bool() after the call. The AMD Triton backend failed to\n'
        '    # compile an int1-typed store into a bool tensor (PassManager::run failed,\n'
        '    # fatal to EngineCore; measured 2026-10-05 on gfx1201).\n'
        '    out_valid_mask_ptr,\n'
        '    out_valid_mask_stride,\n'
        '    sampled_token_ids_ptr,  # [batch_size]\n'
        '    topk_indices_ptr,  # [batch_size, NUM_TOPK] (unused when NUM_TOPK == 0)\n'
        '    topk_indices_stride,\n'
        '    expanded_idx_mapping_ptr,  # [batch_size] -> req_state_idx\n'
        '    num_per_req_token_ids_ptr,  # [max_num_reqs]\n'
        '    per_req_token_ids_ptr,  # [max_num_reqs, MAX_LOGPROB_TOKEN_IDS]\n'
        '    per_req_token_ids_stride,\n'
        '    NUM_TOPK: tl.constexpr,\n'
        '    PADDED_COLS: tl.constexpr,\n'
        '):\n'
        '    batch_idx = tl.program_id(0)\n'
        '\n'
        '    # Column 0: always the sampled token, always valid.\n'
        '    sampled = tl.load(sampled_token_ids_ptr + batch_idx)\n'
        '    tl.store(out_token_ids_ptr + batch_idx * out_token_ids_stride, sampled)\n'
        '    tl.store(out_valid_mask_ptr + batch_idx * out_valid_mask_stride, 1)\n'
        '\n'
        '    req_state_idx = tl.load(expanded_idx_mapping_ptr + batch_idx)\n'
        '    num_custom = tl.load(num_per_req_token_ids_ptr + req_state_idx)\n'
        '\n'
        '    col = tl.arange(0, PADDED_COLS)\n'
        '    tid_base = out_token_ids_ptr + batch_idx * out_token_ids_stride + 1\n'
        '    mask_base = out_valid_mask_ptr + batch_idx * out_valid_mask_stride + 1\n'
        '\n'
        '    # ROCm fix 1: no runtime branch selecting between two pointer bases\n'
        '    # (a branch keyed on the num_custom scalar broke the AMD backend\'s\n'
        '    # make_ttgir). Load BOTH\n'
        '    # sources with their own masks and combine with tl.where. The masked\n'
        '    # loads never touch memory outside their masks, so the topk load is\n'
        '    # inert both on a custom row and when NUM_TOPK == 0 (the caller passes\n'
        '    # an unrelated tensor as the pointer then).\n'
        '    use_custom = num_custom > 0\n'
        '    custom_valid = (col < num_custom) & use_custom\n'
        '    topk_valid = (col < NUM_TOPK) & (num_custom <= 0)\n'
        '    custom = tl.load(\n'
        '        per_req_token_ids_ptr + req_state_idx * per_req_token_ids_stride + col,\n'
        '        mask=custom_valid,\n'
        '        other=0,\n'
        '    )\n'
        '    topk = tl.load(\n'
        '        topk_indices_ptr + batch_idx * topk_indices_stride + col,\n'
        '        mask=topk_valid,\n'
        '        other=0,\n'
        '    )\n'
        '    tokens = tl.where(custom_valid, custom, topk).to(tl.int64)\n'
        '    valid = custom_valid | topk_valid\n'
        '    tl.store(tid_base + col, tokens, mask=valid)\n'
        '    # ROCm fix 2: integer 0/1 into a uint8 buffer, never an int1-typed store\n'
        '    # into a bool tensor.\n'
        '    ones = tl.where(valid, 1, 0).to(tl.int8)\n'
        '    tl.store(mask_base + col, ones, mask=valid)\n'
        '\n'
        '\n'
        'def _fill_logprob_token_ids_torch(\n'
        '    sampled_token_ids: torch.Tensor,\n'
        '    topk_token_ids: torch.Tensor,\n'
        '    num_topk: int,\n'
        '    expanded_idx_mapping: torch.Tensor,\n'
        '    num_per_req_token_ids: torch.Tensor,\n'
        '    per_req_token_ids: torch.Tensor,\n'
        '    num_cols: int,\n'
        ') -> tuple[torch.Tensor, torch.Tensor]:\n'
        '    """Pure-torch twin of _fill_logprob_token_ids_kernel.\n'
        '\n'
        '    Returns `(logprob_token_ids [batch, 1 + num_cols] (same dtype as\n'
        '    sampled_token_ids), valid_mask bool)`. Column 0 is the sampled token and\n'
        '    always valid. For row b with r = expanded_idx_mapping[b] and\n'
        '    n = num_per_req_token_ids[r]: n > 0 fills columns 1..n from\n'
        '    per_req_token_ids[r, :n], otherwise columns 1..num_topk come from\n'
        '    topk_token_ids[b, :num_topk]. Everything else is 0 and invalid. Never\n'
        '    reads topk_token_ids\' data when num_topk == 0 (the caller passes an\n'
        '    unrelated tensor then).\n'
        '    """\n'
        '    batch_size = sampled_token_ids.shape[0]\n'
        '    out = sampled_token_ids.new_zeros((batch_size, 1 + num_cols))\n'
        '    valid_mask = torch.zeros_like(out, dtype=torch.bool)\n'
        '    out[:, 0] = sampled_token_ids\n'
        '    valid_mask[:, 0] = True\n'
        '    for b in range(batch_size):\n'
        '        r = int(expanded_idx_mapping[b])\n'
        '        n = int(num_per_req_token_ids[r])\n'
        '        if n > 0:\n'
        '            out[b, 1 : 1 + n] = per_req_token_ids[r, :n]\n'
        '            valid_mask[b, 1 : 1 + n] = True\n'
        '        elif num_topk > 0:\n'
        '            out[b, 1 : 1 + num_topk] = topk_token_ids[b, :num_topk]\n'
        '            valid_mask[b, 1 : 1 + num_topk] = True\n'
        '    return out, valid_mask\n',
    ),
    (
        '        num_cols = max(num_logprobs, max_per_req_token_ids)\n'
        '        logprob_token_ids = sampled_token_ids.new_zeros((batch_size, 1 + num_cols))\n'
        '        valid_mask = torch.zeros_like(logprob_token_ids, dtype=torch.bool)\n'
        '        _fill_logprob_token_ids_kernel[(batch_size,)](\n'
        '            logprob_token_ids,\n'
        '            logprob_token_ids.stride(0),\n'
        '            valid_mask,\n'
        '            valid_mask.stride(0),\n'
        '            sampled_token_ids,\n'
        '            topk_token_ids,\n'
        '            topk_token_ids.stride(0),\n'
        '            expanded_idx_mapping,\n'
        '            logprob_token_ids_state.num_token_ids.gpu,\n'
        '            logprob_token_ids_state.token_ids.gpu,\n'
        '            logprob_token_ids_state.token_ids.gpu.stride(0),\n'
        '            NUM_TOPK=num_logprobs,\n'
        '            PADDED_COLS=triton.next_power_of_2(num_cols),\n'
        '        )\n',
        # ------------------------------------------------------------------
        '        num_cols = max(num_logprobs, max_per_req_token_ids)\n'
        '        impl = _LOGPROB_TOKEN_IDS_IMPL\n'
        '        if impl not in ("auto", "torch", "triton"):\n'
        '            impl = "auto"\n'
        '        use_kernel = impl == "triton" or (\n'
        '            impl == "auto" and not _TRITON_FILL_FAILED\n'
        '        )\n'
        '        if use_kernel:\n'
        '            logprob_token_ids = sampled_token_ids.new_zeros(\n'
        '                (batch_size, 1 + num_cols))\n'
        '            # uint8, not bool: the AMD Triton backend failed to compile an\n'
        '            # int1-typed store into a bool tensor (PassManager::run failed,\n'
        '            # fatal to EngineCore; measured 2026-10-05). Converted after the call.\n'
        '            valid_mask8 = torch.zeros_like(logprob_token_ids, dtype=torch.uint8)\n'
        '            try:\n'
        '                _fill_logprob_token_ids_kernel[(batch_size,)](\n'
        '                    logprob_token_ids,\n'
        '                    logprob_token_ids.stride(0),\n'
        '                    valid_mask8,\n'
        '                    valid_mask8.stride(0),\n'
        '                    sampled_token_ids,\n'
        '                    topk_token_ids,\n'
        '                    topk_token_ids.stride(0),\n'
        '                    expanded_idx_mapping,\n'
        '                    logprob_token_ids_state.num_token_ids.gpu,\n'
        '                    logprob_token_ids_state.token_ids.gpu,\n'
        '                    logprob_token_ids_state.token_ids.gpu.stride(0),\n'
        '                    NUM_TOPK=num_logprobs,\n'
        '                    PADDED_COLS=triton.next_power_of_2(num_cols),\n'
        '                )\n'
        '                valid_mask = valid_mask8.bool()\n'
        '            except Exception as exc:\n'
        '                if not _TRITON_FILL_FAILED:\n'
        '                    _TRITON_FILL_FAILED.append(True)\n'
        '                    logging.warning(\n'
        '                        "_fill_logprob_token_ids_kernel failed on the AMD "\n'
        '                        "Triton backend (%s: %s); using the pure-torch fill "\n'
        '                        "for the rest of this process",\n'
        '                        type(exc).__name__, exc)\n'
        '                if impl == "triton":\n'
        '                    raise\n'
        '                logprob_token_ids, valid_mask = _fill_logprob_token_ids_torch(\n'
        '                    sampled_token_ids, topk_token_ids, num_logprobs,\n'
        '                    expanded_idx_mapping,\n'
        '                    logprob_token_ids_state.num_token_ids.gpu,\n'
        '                    logprob_token_ids_state.token_ids.gpu,\n'
        '                    num_cols,\n'
        '                )\n'
        '        else:\n'
        '            logprob_token_ids, valid_mask = _fill_logprob_token_ids_torch(\n'
        '                sampled_token_ids, topk_token_ids, num_logprobs,\n'
        '                expanded_idx_mapping,\n'
        '                logprob_token_ids_state.num_token_ids.gpu,\n'
        '                logprob_token_ids_state.token_ids.gpu,\n'
        '                num_cols,\n'
        '            )\n',
    ),
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def patched(text: str) -> str:
    for old, new in REPLACEMENTS:
        count = text.count(old)
        if count != 1:
            raise SystemExit(
                f"anchor not found exactly once ({count}): {old.splitlines()[0]!r}"
            )
        text = text.replace(old, new, 1)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="ROCm-safe vLLM logprob_token_ids fill kernel")
    ap.add_argument("--check", metavar="PATH", help="verify the preimage and dry-run")
    ap.add_argument("--apply", metavar="PATH", help="edit in place and assert the postimage")
    ap.add_argument("--print-hashes", action="store_true")
    args = ap.parse_args(argv)

    if args.print_hashes:
        print(f"preimage  {PREIMAGE_SHA256}")
        print(f"postimage {POSTIMAGE_SHA256}")
        return 0

    path = args.check or args.apply
    if not path:
        ap.error("--check or --apply is required")
    with open(path, encoding="utf-8") as fh:
        original = fh.read()

    observed = sha256_text(original)
    if observed != PREIMAGE_SHA256:
        print(f"REFUSING: preimage sha256 mismatch\n  expected {PREIMAGE_SHA256}\n  observed {observed}",
              file=sys.stderr)
        return 2

    new_text = patched(original)
    result = sha256_text(new_text)
    if result != POSTIMAGE_SHA256:
        print(f"REFUSING: postimage sha256 mismatch\n  expected {POSTIMAGE_SHA256}\n  observed {result}",
              file=sys.stderr)
        return 3

    if args.check:
        print(f"OK preimage  {observed}")
        print(f"OK postimage {result} (dry run, file unchanged)")
        return 0

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new_text)
    print(f"patched {path}\n  preimage  {observed}\n  postimage {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
