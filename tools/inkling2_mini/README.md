# Inkling 2 Mini: Strix Halo reference traces

These are reference outputs for developing the native HIP port, **not** a
TensorFold backend or a test of its kernels. Both examples are short, text-only,
batch-one greedy requests. The IDs include thinking, answer and stop tokens;
matching only the visible answer can hide an earlier divergence. Exact matches
are expected on the pinned reference stack, not necessarily for other kernels.
These cases do not exercise long-context eviction, multimodal input or MTP.

To reproduce, use ROCm PyTorch (tested: 2.13.0+rocm7.2), Accelerate, and
[Transformers at 6173a2f6](https://github.com/huggingface/transformers/commit/6173a2f679ff4ed393afc499aefc569da7e68bc1).
Select the gfx1151 device; allow at least 60 GiB of available graphics memory.
Use the [original checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7)
including config and tokenizer. The optional `mtp.safetensors` is not used.
The main shard has SHA-256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`.

```sh
HF_HUB_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1 python tools/inkling2_mini/check_reference.py /path/to/Inkling-2-Mini
```

The checker removes `quantization_config` from the in-memory config to bypass
ModelOpt quantizer selection and requests BF16, eager attention and eager
experts. The pinned Transformers loader, not this script, dequantizes the
checkpoint's NVFP4 and MXFP8 expert weights to the requested parameter dtype.
No checkpoint files are modified. The original arithmetic run on a Radeon 8060S peaked at 50.2 GiB of
PyTorch-reserved device memory. No GGUF conversion is involved.

To check the committed prompts and token sequences, fixture validation, and
comparison logic without PyTorch, a GPU, or the checkpoint, run from the repo root:

```sh
python3 -m unittest discover -s tools/inkling2_mini -v
```
