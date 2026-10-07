#!/usr/bin/env python3
"""Probe: requested logprob_token_ids on EXPANDED logits, as the speculative rejection sampler calls it.

Run by the orchestrator inside the serve container, as its own process (tiny tensors):

    python3 probe_spec_logprob_token_ids.py

Two requests, three logit rows (request 0 verified two draft positions, request 1 one).
Request 0 asks for ids [7, 9]; request 1 asks for none and gets top-1. Compares
`compute_topk_scores` with a pure-Python log-softmax and prints one JSON line; exit 0
only when every requested id is present with the right value on every row.
"""
from __future__ import annotations

import json
import math
import sys

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState, compute_topk_scores


def main() -> int:
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    vocab = 64
    logits = torch.randn(3, vocab, device=dev, dtype=torch.float32)
    state = LogprobTokenIdsState(4, dev)
    state.add_request(0, SamplingParams(logprobs=1, logprob_token_ids=[7, 9]))
    state.add_request(1, SamplingParams(logprobs=1))
    state.apply_staged_writes()
    expanded = torch.tensor([0, 0, 1], device=dev, dtype=torch.int32)
    sampled = logits.argmax(-1)
    out = compute_topk_scores(logits, 1, sampled, [0, 2, 3], logprob_token_ids_state=state,
                              expanded_idx_mapping=expanded, max_per_req_token_ids=2)
    ids = out.logprob_token_ids.cpu().tolist()
    lps = out.logprobs.cpu().tolist()
    ref = torch.log_softmax(logits.double().cpu(), -1)
    ok = True
    for row in (0, 1):
        got = dict(zip(ids[row][1:], lps[row][1:]))
        for tid in (7, 9):
            ok &= tid in got and math.isclose(got[tid], ref[row, tid].item(), abs_tol=1e-3)
    ok &= ids[2][1] == int(sampled[2]) and math.isclose(lps[2][1], ref[2, ids[2][1]].item(), abs_tol=1e-3)
    ok &= lps[2][2] == float("-inf")
    print(json.dumps({"ok": bool(ok), "shape": list(out.logprob_token_ids.shape), "ids": ids,
                      "cu": out.cu_num_generated_tokens}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
