import os

import gradio as gr
from PIL import Image

from background_replacer import replace_background


def generate(
        image,
        positive_prompt,
        negative_prompt,
        seed,
        depth_map_feather_threshold,
        depth_map_dilation_iterations,
        depth_map_blur_radius,
        manual_caption=None,
        object_prompt=None,
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

    return replace_background(image, positive_prompt, negative_prompt, options, manual_caption, object_prompt)


seed = -1
depth_map_feather_threshold = 128
depth_map_dilation_iterations = 10

depth_map_blur_radius = 10

folder_path = "/home/ilias.papastratis/Downloads/toyotav2/img/1_t0y0tamr2 car/"
folder_path = "/home/ilias.papastratis/workdir/data/formulae/img/10_formulae racecar"
# Use os.listdir to get all files in the directory
files_in_dir = os.listdir(folder_path)

# Filter out any non-image files
image_files = [file for file in files_in_dir if file.endswith(('jpg', 'png', 'jpeg'))]

prompts = [" on a racetrack sunny day",
           "on a racetrack by night",
           " in a studio setting, with a dark background and bright lights.",
           "on a city highway by night"]

# accelerate launch --num_cpu_threads_per_process=2 "./sdxl_train_network.py" --enable_bucket --min_bucket_reso=256 --max_bucket_reso=2048 --pretrained_model_name_or_path="SG161222/RealVisXL_V4.0" --train_data_dir="/home/ilias.papastratis/workdir/data/formulae/img/" --resolution="1024,1024" --output_dir="/home/ilias.papastratis/workdir/data/formulae/model" --logging_dir="/home/ilias.papastratis/workdir/data/formulae/log" --network_alpha="1" --save_model_as=safetensors --network_module=networks.lora --text_encoder_lr=0.0003 --unet_lr=0.0003 --network_dim=128 --output_name="formulae_racecar" --lr_scheduler_num_cycles="20"  --no_half_vae --learning_rate="0.0003" --lr_scheduler="constant_with_warmup" --lr_warmup_steps="172640" --train_batch_size="1" --max_train_steps="1726400" --save_every_n_epochs="1" --mixed_precision="bf16" --save_precision="bf16" --caption_extension=".txt" --cache_latents --optimizer_type="Adafactor" --optimizer_args scale_parameter=False relative_step=False warmup_init=False --max_data_loader_n_workers="0" --bucket_reso_steps=64 --gradient_checkpointing --xformers --log_with wandb --bucket_no_upscale --noise_offset=0.0


for idx, file in enumerate(image_files):

    rendered_img = Image.open(os.path.join(folder_path, file)).convert('RGB')

    view = file.split('blender_')[-1].split('_')[0]
    print(view)
    object_prompt = f"{view} view of formula  racecar"
    generated_img_list = generate(
        rendered_img,
        positive_prompt=prompts[idx % len(prompts)],
        negative_prompt="(octane render, render, drawing, anime, bad photo, bad photography:1.3), (worst quality, low quality, blurry:1.2), (bad teeth, deformed teeth, deformed lips), (bad anatomy, bad proportions:1.1), (deformed iris, deformed pupils), (deformed eyes, bad eyes), (deformed face, ugly face, bad face), (deformed hands, bad hands, fused fingers), morbid, mutilated, mutation, disfigured",
        seed=-1,
        depth_map_feather_threshold=depth_map_feather_threshold,
        depth_map_dilation_iterations=depth_map_dilation_iterations,
        depth_map_blur_radius=depth_map_blur_radius,
        progress=gr.Progress(track_tqdm=True),
        object_prompt=object_prompt
    )

    generated_images = generated_img_list[0]

    for y_idx, new_img in enumerate(generated_images):
        print(type(new_img))
        new_path = f"/home/ilias.papastratis/workdir/data/formulae/img/bcg/prompt_{idx % len(prompts)}_{y_idx}_1_{file}"
        print(new_path)
        new_img.save(new_path)
    #




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

        angle = img_file.split('blender_')[-1].split('_')[0]

        print(angle)
        caption = manual_captions[angle]

        # print(caption)
        caption_file_path = img_file.replace('.png', '.txt')
        print(os.path.join(folder_path, caption_file_path))
        with open(os.path.join(folder_path, caption_file_path), 'w') as f:
            f.write(caption)
        f.close()





1,808 tokens
Response

Markdown
```python
import os
import argparse
from PIL import Image

from background_replacer import replace_background


def parse_args():
"""Parses command line arguments."""

parser = argparse.ArgumentParser(description="Generates images using positive and negative prompts.")

parser.add_argument(
"image_path",
type=str,
help="Path to the input image file.",
)

parser.add_argument(
"positive_prompt",
type=str,
help="Positive prompt for the image generation.",
)

parser.add_argument(
"negative_prompt",
type=str,
help="Negative prompt for the image generation.",
)

parser.add_argument(
"--seed",
type=int,
default=-1,
help="Random seed for the text-to-image generation.",
)

parser.add_argument(
"--depth_map_feather_threshold",
type=int,
default=128,
help="Threshold for feathering the depth map.",
)

parser.add_argument(
"--depth_map_dilation_iterations",
type=int,
default=10,
help="Number of iterations for dilating the depth map.",
)

parser.add_argument(
"--depth_map_blur_radius",
type=int,
default=10,
help="Radius for blurring the depth map.",
)

parser.add_argument(
"--manual_caption",
type=str,
default=None,
help="Manual caption for the generated image.",
)

parser.add_argument(
"--object_prompt",
type=str,
default=None,
help="Object prompt for the image generation.",
)

return parser.parse_args()


def main():
"""Generates images using positive and negative prompts."""

args = parse_args()

image = Image.open(args.image_path).convert("RGB")

generated_img_list = replace_background(
image,
args.positive_prompt,
args.negative_prompt,
seed=args.seed,
depth_map_feather_threshold=args.depth_map_feather_threshold,
depth_map_dilation_iterations=args.depth_map_dilation_iterations,
depth_map_blur_radius=args.depth_map_blur_radius,
manual_caption=args.manual_caption,
object_prompt=args.object_prompt,
)

generated_images = generated_img_list[0]

for y_idx, new_img in enumerate(generated_images):
new_path = f"generated_image_{y_idx}.png"
new_img.save(new_path)


if __name__ == "__main__":
main()
