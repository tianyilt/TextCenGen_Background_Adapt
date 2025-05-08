#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TextCenGen: Attention-Guided Text-Centric Background Adaptation for Text-to-Image Generation
ICML 2025
"""

import os
current_directory = os.getcwd()
import sys
sys.path.append(current_directory)
import argparse
import torch
from diffusers import AutoPipelineForText2Image
from datetime import datetime
import json
from models.replacement import my_attn_processor
from utils.frame import Frame
from utils.guidance_replacement import replace_attn_with_move_object_against_single_point
from utils.ptp_utils import fill_tensor
from utils.vis_utils import save_images
from functools import partial

def extract_substring(s):
    last_slash_index = s.rfind('/')
    if last_slash_index == -1:
        last_slash_index = 0
    else:
        last_slash_index += 1
    dot_index = s.find('.', last_slash_index)
    if dot_index == -1:
        dot_index = len(s)
    return s[last_slash_index:dot_index]


#skip self-attention
def init_model(pipe,prompt,guidance_func,region):
    attn_procs={}
    tokens=pipe.tokenizer.tokenize(prompt)
    for name in pipe.unet.attn_processors.keys():
        if "attn2" in name:
            attn_procs[name]=my_attn_processor(guidance_func,region,len(tokens),place_in_unet=name)
        else:
            attn_procs[name]=pipe.unet.attn_processors[name]
    pipe.unet.set_attn_processor(attn_procs)


parser = argparse.ArgumentParser()
parser.add_argument('--num_inference_steps', type=int, default=50, help='The number of inference steps for pipe')
parser.add_argument('--height', type=int, default=512, help='The height for pipe')
parser.add_argument('--width', type=int, default=512, help='The width for pipe')

parser.add_argument('--move_factor', type=float, default=1, help='The proportion of repulsive movement')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for soft thresholding attention maps')
parser.add_argument('--sharpness', type=float, default=1, help='Sharpness parameter for soft thresholding')
parser.add_argument('--region_exclusion', type=float, default=0.75, help='Region exclusion strength')
parser.add_argument('--theta', type=float, default=0.25, help='Conflict detection threshold')
parser.add_argument('--repulsive_force', type=float, default=14)
parser.add_argument('--margin_force', type=float, default=0.4)


parser.add_argument('--prompt', type=str, default="Photo of a fruit made of feathers with a bee on it.")
parser.add_argument('--position', type=str, default="left")


args = parser.parse_args()
model_id = "runwayml/stable-diffusion-v1-5"
frames = [
	Frame(7, 22, 15, 42, 64),
	Frame(49, 22, 57, 50, 64),
	Frame(22, 7, 42, 15, 64),
	Frame(22, 49, 42, 57, 64),
	Frame(28, 22, 48, 30, 64)
]
positionToFrames={'up':frames[4],'down':frames[1],'left':frames[2],'right':frames[0],'middle':frames[3]}

device="cuda"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = AutoPipelineForText2Image.from_pretrained(model_id, 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True
).to("cuda")

guidance_func = partial(replace_attn_with_move_object_against_single_point,f_repl=args.repulsive_force,f_margin=args.margin_force,clamp=args.move_factor,threshold=args.threshold,sharpness=args.sharpness,region_exclusion=args.region_exclusion,theta=args.theta)

seed = 56775
generator = torch.manual_seed(seed)
negative_prompt = "monocolor, monotony, cartoon style, many texts, pure cloud, pure sea, extra texts, texts, monochrome, flattened, lowres, longbody, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"



prompt = args.prompt
frame=frames[0]
if args.position in positionToFrames:
    frame=positionToFrames[args.position]
region=fill_tensor(frame.x, frame.y, frame.a, frame.b, frame.deep_h, frame.deep_w).to(device)
init_model(pipe,prompt,guidance_func,region)
image = pipe(prompt, height=args.height, width=args.width,
                    num_inference_steps=args.num_inference_steps, generator=generator,
                    negative_prompt=negative_prompt).images[0]
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
outdir=extract_substring(args.path)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_path = ('{}/output/{}/{}_{}').format(current_directory, outdir,prompt[:50], current_time)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
input_params = {
    'prompt': prompt,
    'seed': seed,
}
data_to_save = {'x': frame.x, 'y': frame.y, 'a': frame.a, 'b': frame.b, 'prompt': prompt, 'seed': seed}
with open(folder_path + '/data.json', 'w') as file:
    json.dump(data_to_save, file, indent=4)
left_top = (frame.y * 8, frame.x * 8)
right_bottom = (frame.b * 8, frame.a * 8)
save_images([image], folder=folder_path, input_params=input_params,
                    titles=['edited'],left_top=left_top,
                right_bottom=right_bottom,save_with_draw_frame=True,save_combined=False)
