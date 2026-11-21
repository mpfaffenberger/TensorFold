"""Replay short greedy text traces on a gfx1151 ROCm device, entirely offline."""
import argparse
import json
from pathlib import Path
from typing import TypedDict


class ReferenceCase(TypedDict):
    prompt: str
    token_ids: list[int]


def load_cases(path: Path) -> list[ReferenceCase]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list) or not data:
        raise ValueError('expected a non-empty list of reference cases')
    prompts: set[str] = set()
    cases: list[ReferenceCase] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f'case {index}: expected an object')
        prompt = item.get('prompt')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f'case {index}: expected a non-empty prompt')
        if prompt in prompts:
            raise ValueError(f'case {index}: duplicate prompt: {prompt!r}')
        ids = item.get('token_ids')
        if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError(f'case {index}: expected a non-empty token_ids list of non-negative integers')
        prompts.add(prompt)
        cases.append({'prompt': prompt, 'token_ids': ids})
    return cases


def compare_case(ids: list[int], case: ReferenceCase) -> None:
    if ids != case['token_ids']:
        raise ValueError(f"{case['prompt']}: expected {case['token_ids']}, generated {ids}")


def check_loading_info(info: dict[str, object]) -> None:
    keys = ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')
    if errors := {key: info[key] for key in keys if info.get(key)}:
        raise RuntimeError(f'Incomplete checkpoint load: {errors}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    cases = load_cases(Path(__file__).with_name('gfx1151-reference.json'))

    import torch
    from transformers import AutoConfig, AutoTokenizer, Inkling2LoopForConditionalGeneration

    if not torch.version.hip or not torch.cuda.is_available():
        parser.error('requires ROCm PyTorch with an available HIP device')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx1151':
        parser.error('device 0 is not gfx1151; select the Radeon via ROCR_VISIBLE_DEVICES')
    cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    if cfg.model_type != 'inkling2_loop' or cfg.quantization_config['quant_algo'] != 'NVFP4':
        parser.error('expected the original mixed-precision Inkling 2 Mini checkpoint')
    # Bypass generic quantizer selection; request BF16 from the pinned loader below.
    del cfg.quantization_config
    tok = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model, info = Inkling2LoopForConditionalGeneration.from_pretrained(
        args.checkpoint, config=cfg, dtype=torch.bfloat16, device_map={'': 'cuda:0'},
        attn_implementation='eager', experts_implementation='eager',
        output_loading_info=True, local_files_only=True,
    )
    check_loading_info(info)
    with torch.inference_mode():
        for case in cases:
            batch = tok.apply_chat_template(
                [{'role': 'user', 'content': case['prompt']}], reasoning_effort=0,
                add_generation_prompt=True, tokenize=True, return_tensors='pt', return_dict=True,
            ).to('cuda:0')
            outputs = model.generate(**batch, do_sample=False, use_cache=True, max_new_tokens=96)
            ids = outputs[0, batch.input_ids.shape[1]:].tolist()
            compare_case(ids, case)
            print(f"PASS {len(ids)} tokens: {case['prompt']}")


if __name__ == '__main__':
    main()
