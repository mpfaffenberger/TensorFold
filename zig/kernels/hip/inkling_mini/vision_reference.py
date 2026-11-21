"""Generate vision-test fixtures with the pinned local transformers (6173a2f6)."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import Inkling2ImageProcessor, Inkling2VisionConfig, Inkling2VisionModel

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkpoint', type=Path)
parser.add_argument('out', type=Path)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
y, x = np.indices((57, 93))
image = Image.fromarray(np.stack([9*x+y, x+7*y, x*y, 3*x+11*y], -1).astype('uint8'))
image.save(args.out / 'reference.png')
processor = Inkling2ImageProcessor.from_pretrained(args.checkpoint)
inputs = processor(images=image, return_tensors='pt')
assert inputs.image_grid_thw.tolist() == [[1, 2, 3]]
config = Inkling2VisionConfig.from_pretrained(args.checkpoint)
config._attn_implementation = 'eager'
torch.set_num_threads(6)
with torch.device('meta'):
    tower = Inkling2VisionModel(config)
# Regenerate the nonpersistent inverse frequency buffer outside the meta context.
tower.rotary_emb = type(tower.rotary_emb)(config)
renames = {
    'embeddings.position_embedding': 'embeddings.position_embedding.weight',
    'ln_pre': 'pre_layernorm', 'ln_1': 'layernorm_before', 'ln_2': 'layernorm_after',
    'ls_1.gamma': 'lambda_1', 'ls_2.gamma': 'lambda_2', 'mlp.c_fc': 'mlp.fc1',
    'mlp.c_proj': 'mlp.fc2', 'proj_in': 'merger.linear_1', 'proj_out': 'merger.linear_2',
}
params = {}
for shard in args.checkpoint.glob('*.safetensors'):
    with safe_open(shard, framework='pt') as sf:
        for key in sf.keys():
            if not key.startswith('model.visual.'):
                continue
            name = key.removeprefix('model.visual.').removeprefix('backbone.')
            for source, target in renames.items():
                name = name.replace(source, target)
            params[name] = sf.get_tensor(key).float()
tower.load_state_dict(params, strict=True, assign=True)
with torch.inference_mode():
    tower.eval()(**inputs).pooler_output.numpy().tofile(args.out / 'reference.f32')
