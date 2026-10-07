#!/usr/bin/env python3
"""Give Qwen3_5's vLLM model an lm_head LoRA target (decision-head adapters).

TARGET (mounted over the container's
    <site-packages>/vllm/model_executor/models/qwen3_5.py:ro
by qwen38-27b-mxfp4-r9700-serve.sh):
    vllm_qwen3_5.py
    Preimage  sha256 29a9e4c1d4ac5158b1981a526020ea6ddfe3d0fa0ac70c91c018f273ffc4ec96
    Postimage sha256 (asserted after --apply; see POSTIMAGE_SHA256)

WHY (measured 2026-10-05): AutoTrust JEV-27B ships its decision head as a LoRA on
`lm_head`. the bench's vLLM 0.27.1 has the machinery (`LogitsProcessorWithLoRA`;
`lora/utils.py:get_supported_lora_modules` adds every name in a module's
`embedding_modules` attribute; `lora_model_runner_mixin.py` passes the TOP-LEVEL
`model.embedding_modules` to the manager). But `Qwen3_5ForCausalLMBase` and
`Qwen3_5ForConditionalGeneration` declare no `lm_head` there, so loading failed with:

    ValueError: expected target modules in {...} but received
    ['language_model.lm_head', ...]

Other models declare it the same way (granite.py):
    embedding_modules = {"embed_tokens": "input_embeddings", "lm_head": "output_embeddings"}

Usage: patch_vllm_qwen3_5_lm_head_lora.py --check | --apply
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PREIMAGE = HERE / "vllm_qwen3_5.py.preimage"
DEFAULT_OUTPUT = HERE / "vllm_qwen3_5.py"

PREIMAGE_SHA256 = "29a9e4c1d4ac5158b1981a526020ea6ddfe3d0fa0ac70c91c018f273ffc4ec96"
POSTIMAGE_SHA256 = "845cdcfdbadf54e15425c8e2cd0931b328cb80939d556f5d381b266a9ff5310f"

COMMENT = "# horde: lm_head is a LoRA target (decision-head adapters, e.g. AutoTrust JEV-27B).\n"
ATTRIBUTE = '    embedding_modules = {"lm_head": "output_embeddings"}\n'

ANCHORS: list[tuple[str, str]] = [
    (
        "class Qwen3_5ForCausalLMBase(\n"
        "    nn.Module,\n"
        "    HasInnerState,\n"
        "    IsHybrid,\n"
        "    SupportsEagle3,\n"
        "    SupportsLoRA,\n"
        "    SupportsPP,\n"
        "):\n"
        "    packed_modules_mapping = {\n",
        "class Qwen3_5ForCausalLMBase(\n"
        "    nn.Module,\n"
        "    HasInnerState,\n"
        "    IsHybrid,\n"
        "    SupportsEagle3,\n"
        "    SupportsLoRA,\n"
        "    SupportsPP,\n"
        "):\n"
        + COMMENT + ATTRIBUTE + "\n"
        "    packed_modules_mapping = {\n",
    ),
    (
        "class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):\n"
        "    supports_multimodal_pruning = True\n\n"
        "    packed_modules_mapping = Qwen3VLForConditionalGeneration.packed_modules_mapping | {\n",
        "class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):\n"
        "    supports_multimodal_pruning = True\n\n"
        + COMMENT + ATTRIBUTE + "\n"
        "    packed_modules_mapping = Qwen3VLForConditionalGeneration.packed_modules_mapping | {\n",
    ),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patched(text: str) -> str:
    for old, new in ANCHORS:
        count = text.count(old)
        if count != 1:
            raise SystemExit(
                f"anchor not found exactly once ({count}): {old.splitlines()[0]!r}"
            )
        text = text.replace(old, new, 1)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="lm_head LoRA target for Qwen3_5 (decision-head adapters)")
    ap.add_argument("--check", action="store_true",
                    help="verify the preimage hash and dry-run; write nothing")
    ap.add_argument("--apply", action="store_true",
                    help="derive the output from the preimage and assert its hash")
    args = ap.parse_args(argv)
    if args.check == args.apply:
        ap.error("exactly one of --check / --apply is required")

    pre = DEFAULT_PREIMAGE.read_bytes()
    actual = sha256_bytes(pre)
    if actual != PREIMAGE_SHA256:
        raise SystemExit(
            f"preimage sha256 mismatch: expected {PREIMAGE_SHA256}, got {actual}")

    derived = patched(pre.decode("utf-8")).encode("utf-8")

    if args.check:
        if DEFAULT_OUTPUT.exists():
            out_sha = sha256_bytes(DEFAULT_OUTPUT.read_bytes())
            status = "matches" if out_sha == sha256_bytes(derived) else "DIFFERS from"
            print(f"--check: preimage ok; {DEFAULT_OUTPUT} {status} the derivation")
        else:
            print(f"--check: preimage ok; {DEFAULT_OUTPUT} does not exist yet")
        return 0

    derived_sha = sha256_bytes(derived)
    if derived_sha != POSTIMAGE_SHA256:
        raise SystemExit(
            f"derived sha256 mismatch: expected {POSTIMAGE_SHA256}, got {derived_sha}")
    DEFAULT_OUTPUT.write_bytes(derived)
    print(f"applied: wrote {DEFAULT_OUTPUT} sha256 {derived_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
