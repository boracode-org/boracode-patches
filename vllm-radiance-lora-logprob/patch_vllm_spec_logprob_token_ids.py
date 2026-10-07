#!/usr/bin/env python3
"""Make vLLM's `logprob_token_ids` work under speculative decoding (DFlash, MTP).

TARGET (mounted over vllm/v1/worker/gpu/spec_decode/rejection_sampler.py):
    vllm_spec_rejection_sampler.py
    Preimage  sha256 e89e00b8688f47ef4c90db3402a27dbe80ff71554d8764766d52e3a50d9d8799

Usage: patch_vllm_spec_logprob_token_ids.py --check <path> | --apply <path> | --print-hashes

WHY (measured 2026-10-05 on the bench: vLLM 0.27.1, DFlash speculative decoding): a request
with `logprob_token_ids` returned the requested ids for the FIRST token only -- that
step goes through `Sampler.__call__`, which hands `compute_topk_scores` the per-request
token-id state. Every later token is verified by `RejectionSampler`, whose
`_get_logprobs_tensors` called `compute_topk_scores` WITHOUT that state and returned
`None` whenever `logprobs` alone would. Result: max_tokens=1 worked, max_tokens>=2
failed every time with HTTP 500 "list index out of range" (the API server indexing a
logprob list shorter than the token list). Upstream's own note in a decision server:
"logprob_token_ids does not [work with speculative decoding] in this vLLM build".

FIX: the rejection sampler mirrors `Sampler.__call__` exactly -- it computes the
batch's widest requested-id count once, returns logprobs when either `logprobs` or
requested ids are present, and passes `logprob_token_ids_state`, the chunk's
`expanded_idx_mapping` and that width to `compute_topk_scores`. The width is the
batch's, not the chunk's, so chunked verification still concatenates.
"""

from __future__ import annotations

import argparse
import hashlib
import sys

PREIMAGE_SHA256 = "e89e00b8688f47ef4c90db3402a27dbe80ff71554d8764766d52e3a50d9d8799"
POSTIMAGE_SHA256 = "7029b63700c8d68be608a3d4ec904ba68a0cfd0bd20955065b0a3728a60ab0d2"

REPLACEMENTS: list[tuple[str, str]] = [
    (
        "        cu_num_logits_np: np.ndarray,\n"
        "        max_num_logprobs: int,\n"
        "    ) -> LogprobsTensors | None:\n"
        "        if max_num_logprobs == NO_LOGPROBS:\n"
        "            return None\n",
        "        cu_num_logits_np: np.ndarray,\n"
        "        max_num_logprobs: int,\n"
        "        expanded_idx_mapping: torch.Tensor,\n"
        "        max_per_req_token_ids: int,\n"
        "    ) -> LogprobsTensors | None:\n"
        "        # Mirrors Sampler.__call__: requested logprob_token_ids alone are enough.\n"
        "        if max_num_logprobs == NO_LOGPROBS and max_per_req_token_ids == 0:\n"
        "            return None\n"
        "        num_logprobs = max_num_logprobs if max_num_logprobs != NO_LOGPROBS else 0\n",
    ),
    (
        "        return compute_topk_scores(\n"
        "            logits,\n"
        "            max_num_logprobs,\n"
        "            flat_sampled,\n"
        "            cu_num_logits_np.tolist() if expanded_logits else None,\n",
        "        return compute_topk_scores(\n"
        "            logits,\n"
        "            num_logprobs,\n"
        "            flat_sampled,\n"
        "            cu_num_logits_np.tolist() if expanded_logits else None,\n"
        "            logprob_token_ids_state=self.sampler.logprob_token_ids_state,\n"
        "            expanded_idx_mapping=expanded_idx_mapping,\n"
        "            max_per_req_token_ids=max_per_req_token_ids,\n",
    ),
    (
        "        max_chunk_logits: int,\n"
        "        max_num_logprobs: int,\n"
        "    ) -> tuple[torch.Tensor, torch.Tensor, LogprobsTensors | None]:\n",
        "        max_chunk_logits: int,\n"
        "        max_num_logprobs: int,\n"
        "        max_per_req_token_ids: int = 0,\n"
        "    ) -> tuple[torch.Tensor, torch.Tensor, LogprobsTensors | None]:\n",
    ),
    (
        "                chunk_cu_num_logits_np,\n"
        "                max_num_logprobs,\n"
        "            )\n",
        "                chunk_cu_num_logits_np,\n"
        "                max_num_logprobs,\n"
        "                input_batch.expanded_idx_mapping[lo:hi],\n"
        "                max_per_req_token_ids,\n"
        "            )\n",
    ),
    (
        "        max_chunk_logits = max(1, MAX_CHUNK_BYTES // (logits.shape[1] * _FP32_BYTES))\n",
        "        # Batch-wide (not per chunk) so chunked verification concatenates.\n"
        "        max_per_req_token_ids = self.sampler.logprob_token_ids_state.max_num_token_ids(\n"
        "            input_batch.idx_mapping_np\n"
        "        )\n"
        "        max_chunk_logits = max(1, MAX_CHUNK_BYTES // (logits.shape[1] * _FP32_BYTES))\n",
    ),
    (
        "            max_chunk_logits,\n"
        "            max_num_logprobs,\n"
        "        )\n",
        "            max_chunk_logits,\n"
        "            max_num_logprobs,\n"
        "            max_per_req_token_ids,\n"
        "        )\n",
    ),
]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def derive(source: str) -> str:
    out = source
    for old, new in REPLACEMENTS:
        if out.count(old) != 1:
            raise SystemExit(f"anchor matched {out.count(old)} times (need exactly 1): {old[:60]!r}")
        out = out.replace(old, new)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="logprob_token_ids under speculative decoding")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", metavar="PATH", help="verify the preimage and dry-run")
    group.add_argument("--apply", metavar="PATH", help="edit in place and assert the postimage")
    group.add_argument("--print-hashes", action="store_true")
    args = parser.parse_args()
    if args.print_hashes:
        print(f"preimage  {PREIMAGE_SHA256}\npostimage {POSTIMAGE_SHA256}")
        return 0
    path = args.check or args.apply
    data = open(path, "rb").read()
    if _sha(data) == POSTIMAGE_SHA256:
        print("already patched")
        return 0
    if _sha(data) != PREIMAGE_SHA256:
        print(f"refusing: {path} is neither the pinned preimage nor the postimage", file=sys.stderr)
        return 2
    derived = derive(data.decode()).encode()
    if _sha(derived) != POSTIMAGE_SHA256:
        print(f"refusing: derived sha256 {_sha(derived)} is not the pinned postimage", file=sys.stderr)
        return 3
    if args.apply:
        open(path, "wb").write(derived)
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
