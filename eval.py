
import pprint
from typing import List

import pyrallis
import torch
from PIL import Image
from config import RunConfig
from pipeline.gligen_pipeline_boxdiff import BoxDiffPipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore

import numpy as np
from utils.drawer import draw_rectangle, DashedImageDraw

import warnings
import json, os
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning)


def load_model():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    stable_diffusion_version = "gligen/diffusers-generation-text-box"
    # If you cannot access the huggingface on your server, you can use the local prepared one.
    # stable_diffusion_version = "../../packages/diffusers/gligen_ckpts/diffusers-generation-text-box"
    stable = BoxDiffPipeline.from_pretrained(stable_diffusion_version, revision="fp16", torch_dtype=torch.float16).to(device)

    return stable


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def run_on_prompt(prompt: List[str],
                  model: BoxDiffPipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config: RunConfig) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)

    gligen_boxes = []
    for i in range(len(config.bbox)):
        x1, y1, x2, y2 = config.bbox[i]
        gligen_boxes.append([x1/512, y1/512, x2/512, y2/512])

    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    gligen_phrases=config.gligen_phrases,
                    gligen_boxes=gligen_boxes,
                    gligen_scheduled_sampling_beta=0.3,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1,
                    bbox=config.bbox,
                    config=config)
    image = outputs.images[0]
    return image


@pyrallis.wrap()
def main(config: RunConfig):
    stable = load_model()

    # read bbox from the pre-prepared .json file
    with open('docs/bbox_as_condition.json', 'r', encoding='utf8') as fp:
        bbox_json = json.load(fp)

    idx = np.arange(len(bbox_json))
    split_idx = list(np.array_split(idx, config.n_splits)[config.which_one - 1])

    for bidx in tqdm(split_idx):

        filename = bbox_json[bidx]['filename']
        sub_dir = filename.split('/')[-2]
        img_name = filename.split('/')[-1].split('.')[0]

        objects = bbox_json[bidx]['objects']
        cls_name = list(objects.keys())
        config.bbox = list(objects.values())
        # import ipdb
        # ipdb.set_trace()
        text_prompt = ''
        gligen_phrases = []
        token_indices = []
        for nidx, n in enumerate(cls_name):
            text_prompt += f'a {n} and '
            if nidx == 0:
                token_indices.append(2)
            else:
                token_indices.append(5 + (nidx - 1) * 3)
            gligen_phrases.append('a {n}')
        config.prompt = text_prompt[:-5]
        config.gligen_phrases = gligen_phrases

        for seed in config.seeds:
            print(f"Current seed is : {seed}")
            g = torch.Generator('cuda').manual_seed(seed)
            controller = AttentionStore()
            controller.num_uncond_att_layers = -16
            image = run_on_prompt(prompt=config.prompt,
                                  model=stable,
                                  controller=controller,
                                  token_indices=token_indices,
                                  seed=g,
                                  config=config)

            prompt_output_path = config.eval_output_path / sub_dir
            prompt_output_path.mkdir(exist_ok=True, parents=True)
            if os.path.isfile(prompt_output_path / f'{img_name}_{seed}.png'):
                continue

            image.save(prompt_output_path / f'{img_name}_{seed}.png')

            canvas = Image.fromarray(np.zeros((image.size[0], image.size[0], 3), dtype=np.uint8) + 220)
            draw = DashedImageDraw(canvas)

            for i in range(len(config.bbox)):
                x1, y1, x2, y2 = config.bbox[i]
                draw.dashed_rectangle([(x1, y1), (x2, y2)], dash=(5, 5), outline=config.color[i], width=5)
            canvas.save(prompt_output_path / f'{img_name}_{seed}_canvas.png')


if __name__ == '__main__':
    main()
