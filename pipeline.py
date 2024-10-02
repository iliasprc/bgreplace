import torch
from diffusers import StableDiffusionXLControlNetPipeline, \
ControlNetModel, AutoencoderKL, UniPCMultistepScheduler,EulerAncestralDiscreteScheduler

device = 'cuda'
pipe = None



def init(use_bria = True):
    global device, pipe


    # "/home/ilias.papastratis/workdir/bria_models/bria_bg_gen_controlnet"
    
    if use_bria:
        depth_cpkt = '/home/ilias.papastratis/workdir/bgreplace/depth_controlnet_test'
        base_model = "/home/ilias.papastratis/workdir/bria_models/BRIA_2.3"
    
        
    else:
        depth_cpkt = "diffusers/controlnet-depth-sdxl-1.0"
        base_model = "SG161222/RealVisXL_V4.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing depth ControlNet...")

    depth_controlnet = ControlNetModel.from_pretrained(
        depth_cpkt,
        torch_dtype=torch.float16
    ).to(device)

    print("Initializing autoencoder...")

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    ).to(device)

    print("Initializing SDXL pipeline...")

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model,
        controlnet=[depth_controlnet],
        vae=vae,
        torch_dtype=torch.float16
        # low_cpu_mem_usage=True
    ).to(device)


    pipe.scheduler = EulerAncestralDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    steps_offset=1
)
    # remove following line if xformers is not installed
    #pipe.enable_xformers_memory_efficient_attention()


def run_pipeline(image, positive_prompt, negative_prompt, seed, lora_model=None,controlnet_conditioning_scale=0.65,num_inference_steps=30):
    pipe.unload_lora_weights()
    if seed == -1:
        print("Using random seed")
        generator = None
    else:
        print("Using seed:", seed)
        generator = torch.manual_seed(seed)
    if lora_model!=None:
        pipe.load_lora_weights(lora_model)
    # if '0%' in positive_prompt:
    #     pipe.load_lora_weights('/home/ilias.papastratis/workdir/trained_models/lora_models/lora_fage_realvisxl_fage0_v3/pytorch_lora_weights.safetensors')
    # elif '5%' in positive_prompt:
    #     pipe.load_lora_weights('/home/ilias.papastratis/workdir/trained_models/lora_models/lora_fage_realvisxl_5fat/checkpoint-2000/pytorch_lora_weights.safetensors')
    images = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        guidance_scale=7,
        generator=generator,
        image=image
    ).images

    return images
