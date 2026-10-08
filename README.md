# boracode patches

Patch scripts and probes that accompany posts on https://boracode.ai/blog. Each folder holds the files for one post and states what it was measured against.

| Folder | Post | Target |
|---|---|---|
| `vllm-radiance-lora-logprob/` | Per-request LoRA swapping on AMD | vLLM 0.27.1 (vLLM-Radiance build), ROCm, gfx1201 |

## How the vLLM scripts work

Each `patch_*.py` is written to be mounted read-only over one file inside a container, so the image is not rebuilt. Every script pins the SHA-256 of the file it expects and refuses anything else, so a changed vLLM stops the script instead of being patched blindly.

- `patch_vllm_logprob_token_ids_rocm.py` and `patch_vllm_spec_logprob_token_ids.py`: `--check <path>` verifies the unpatched file, `--apply <path>` writes the patched file and verifies the result, `--print-hashes` prints the expected pre- and post-image hashes.
- `patch_vllm_qwen3_5_lm_head_lora.py`: `--check` or `--apply`, with no path argument; read its header for the file it derives.

The `probe_*.py` scripts send one request that exercises the patched path and print what came back.

## How these were written

The scripts in this repository were written with an AI coding assistant (Claude), directed by the author, and run on the author's own hardware. The results reported in the posts come from those runs. They are published here as a record of what was run, not as submissions to any upstream project. Check a project's contribution policy on AI-assisted code before reusing them in a pull request.

## Licence

Apache License 2.0, the same licence as vLLM, which these scripts patch. The full text is in `LICENSE`.

## Provenance

These patch vLLM (Apache-2.0) and are intended for upstream discussion. The adapter used in the post is AutoTrust's JEV-27B (Apache-2.0); the ROCm build is StillDeadcode's vLLM-Radiance.
