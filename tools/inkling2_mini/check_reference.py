"""Replay short greedy text traces on a gfx1151 ROCm device, entirely offline."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, Inkling2LoopForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    if not torch.version.hip or not torch.cuda.is_available():
        parser.error('requires ROCm PyTorch with an available HIP device')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx1151':
        parser.error('device 0 is not gfx1151; select the Radeon via ROCR_VISIBLE_DEVICES')
    cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    if cfg.model_type != 'inkling2_loop' or cfg.quantization_config['quant_algo'] != 'NVFP4':
        parser.error('expected the original mixed-precision Inkling 2 Mini checkpoint')
    # Use the Inkling-specific BF16 checkpoint transform, not generic ModelOpt kernels.
    del cfg.quantization_config
    tok = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model, info = Inkling2LoopForConditionalGeneration.from_pretrained(
        args.checkpoint, config=cfg, dtype=torch.bfloat16, device_map={'': 'cuda:0'},
        attn_implementation='eager', experts_implementation='eager',
        output_loading_info=True, local_files_only=True,
    )
    if any(info.values()):
        raise RuntimeError(f'Incomplete checkpoint load: {info}')
    cases = json.loads(Path(__file__).with_name('gfx1151-reference.json').read_text())
    with torch.inference_mode():
        for case in cases:
            batch = tok.apply_chat_template(
                [{'role': 'user', 'content': case['prompt']}], reasoning_effort=0,
                add_generation_prompt=True, tokenize=True, return_tensors='pt', return_dict=True,
            ).to('cuda:0')
            outputs = model.generate(**batch, do_sample=False, use_cache=True, max_new_tokens=96)
            ids = outputs[0, batch.input_ids.shape[1]:].tolist()
            if ids != case['token_ids']:
                raise ValueError(f"{case['prompt']}: generated IDs differ: {ids}")
            print(f"PASS {len(ids)} tokens: {case['prompt']}")


if __name__ == '__main__':
    main()
