import os
import random

import gradio as gr

from background_replacer import replace_background

import os
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting
from fastapi import File, UploadFile, FastAPI, Depends, HTTPException

from pydantic import BaseModel, ValidationError,  Field









def generate(
    image,
    positive_prompt,
    negative_prompt,
    seed,
    depth_map_feather_threshold,
    depth_map_dilation_iterations,
    depth_map_blur_radius,
    manual_caption=None,
    object_prompt = None,
    progress=gr.Progress(track_tqdm=True)
):
    if image is None:
        return [None, None, None, None]

    options = {
        'seed': seed,
        'depth_map_feather_threshold': depth_map_feather_threshold,
        'depth_map_dilation_iterations': depth_map_dilation_iterations,
        'depth_map_blur_radius': depth_map_blur_radius,
    }

    return replace_background(image, positive_prompt, negative_prompt, options,manual_caption,object_prompt)


seed =  -1
depth_map_feather_threshold = 128
depth_map_dilation_iterations = 50

depth_map_blur_radius = 30

folder_path = "/home/ilias.papastratis/Downloads/toyotav2/img/1_t0y0tamr2 car/"
folder_path = "/home/ilias.papastratis/workdir/data/formulae/img/5_formulae racecar"
# Use os.listdir to get all files in the directory
files_in_dir = os.listdir(folder_path)

# Filter out any non-image files
image_files = [file for file in files_in_dir if file.endswith(('jpg', 'png', 'jpeg'))]

prompts = [

    " parked on a gravelly overlook with a view of a marina and mountains, (two individuals by the guardrail:1.1),photorealistic, ",
    " parked on the road on a sunny day, trees alongside the road,photorealistic,  ",
    " parked in an asphalt road in front of a  dense forest landscape with a narrow path and lush greenery.photorealistic, ",

    " under a blue sky with a green mountain range in the background,photorealistic, .",
    " next to  a charging station outside a rustic, eco-friendly wooden house with a glass roof  in a mountainous region with gravel ground.photorealistic, ",
    " parked   in front of a  panoramic view of a peaceful valley with a river flowing through it.photorealistic, ",
    " parked in front of a  captivating view of a desert with towering sand dunes under a scorching sun.photorealistic, ",
    "driving on a dirt road  with a snow mountain range in the background, conveying a sense of adventure and capability in various terrains, a snowy day,photorealistic, "

]
prompts = [" on a racetrack sunny day",
            "on a ractrack by night"
           "on a photo studio",
           "on city by night"]


#accelerate launch --num_cpu_threads_per_process=2 "./sdxl_train_network.py" --enable_bucket --min_bucket_reso=256 --max_bucket_reso=2048 --pretrained_model_name_or_path="SG161222/RealVisXL_V4.0" --train_data_dir="/home/ilias.papastratis/workdir/data/formulae/img/" --resolution="1024,1024" --output_dir="/home/ilias.papastratis/workdir/data/formulae/model" --logging_dir="/home/ilias.papastratis/workdir/data/formulae/log" --network_alpha="1" --save_model_as=safetensors --network_module=networks.lora --text_encoder_lr=0.0003 --unet_lr=0.0003 --network_dim=128 --output_name="formulae_racecar" --lr_scheduler_num_cycles="20"  --no_half_vae --learning_rate="0.0003" --lr_scheduler="constant_with_warmup" --lr_warmup_steps="172640" --train_batch_size="1" --max_train_steps="1726400" --save_every_n_epochs="1" --mixed_precision="bf16" --save_precision="bf16" --caption_extension=".txt" --cache_latents --optimizer_type="Adafactor" --optimizer_args scale_parameter=False relative_step=False warmup_init=False --max_data_loader_n_workers="0" --bucket_reso_steps=64 --gradient_checkpointing --xformers --log_with wandb --bucket_no_upscale --noise_offset=0.0
random.shuffle(image_files)
image_files = image_files[:20]

for idx,file in enumerate(image_files):
    rendered_img = Image.open(os.path.join(folder_path, file)).convert('RGB')

    view =  file.split('blender_')[-1].split('_')[0]
    for depth in range(1,50,5):

        print(view)
        object_prompt = f"{view} view of formula  racecar with a  blue and black metallic color with white logos, a  rear wing and a diffuser."
        generated_img_list = generate(
            rendered_img,
            positive_prompt=prompts[idx%len(prompts)],
            negative_prompt="(octane render, render, drawing, anime, bad photo, bad photography:1.3), (worst quality, low quality, blurry:1.2), (bad teeth, deformed teeth, deformed lips), (bad anatomy, bad proportions:1.1), (deformed iris, deformed pupils), (deformed eyes, bad eyes), (deformed face, ugly face, bad face), (deformed hands, bad hands, fused fingers), morbid, mutilated, mutation, disfigured",
            seed=-1,
            depth_map_feather_threshold=depth_map_feather_threshold,
            depth_map_dilation_iterations=depth,
            depth_map_blur_radius=depth,
            progress=gr.Progress(track_tqdm=True),
            object_prompt=object_prompt
        )

        generated_images = generated_img_list[0]
        pre_processing_images = generated_img_list[2]
        for y_idx,new_img in enumerate(generated_images):
            print(type(new_img))
            new_path = f"/home/ilias.papastratis/workdir/data/ford_3d/formula/test/depth_{depth}prompt_{idx%len(prompts)}_{y_idx}_1_{file}"
            print(new_path)
            new_img.save(new_path)
        #
        for y_idx,new_img in enumerate(pre_processing_images):
            print(type(new_img))
            new_path = f"/home/ilias.papastratis/workdir/data/ford_3d/formula/test/depth_{depth}_{new_img[-1]}_{idx%len(prompts)}_{y_idx}_1_{file}"
            new_img[0].save(new_path)



exit()
def manuan_captioning(prompts):
    folder_path = "/home/ilias.papastratis/Downloads/toyota_bg3/"
    # Use os.listdir to get all files in the directory
    files_in_dir = os.listdir(folder_path)

    # Filter out any non-image files
    image_files = [file for file in files_in_dir if file.endswith(('jpg', 'png', 'jpeg'))]
    for img_file in image_files:
        print(img_file)

older_path = "/home/ilias.papastratis/workdir/data/ford_3d/img/5_fordexplorerev car/"
def manual_captioning_blender_images(prompts):
    POSITIVE_PROMPT_PREFIX = "Raw realistic photo,"
    POSITIVE_PROMPT_SUFFIX = "masterpiece,4K,HD,commercial product photography, 24mm lens f/8"
    folder_path = "/home/ilias.papastratis/workdir/data/ford_3d/img/5_fordexplorerev car/"
    # Use os.listdir to get all files in the directory
    files_in_dir = os.listdir(folder_path)

    # Filter out any non-image files
    image_files = [file for file in files_in_dir if file.endswith(('jpg', 'png', 'jpeg'))]
    for img_file in image_files:
        print(img_file)

        angle =  img_file.split('blender_')[-1].split('_')[0]

        print(  angle)
        caption = manual_captions[angle]

        # print(caption)
        caption_file_path = img_file.replace('.png','.txt')
        print(os.path.join(folder_path,caption_file_path))
        with open(os.path.join(folder_path,caption_file_path),'w') as f:

            f.write(caption)
        f.close()
manual_captioning_blender_images(prompts)
