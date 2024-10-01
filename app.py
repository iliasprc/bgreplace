import os

import gradio as gr

from background_replacer import replace_background

developer_mode = os.getenv('DEV_MODE', True)

from captioner import init as init_captioner
from upscaler import init as init_upscaler
from segmenter import init as init_segmenter
from depth_estimator import init as init_depth_estimator
from pipeline import init as init_pipeline


init_upscaler()
init_segmenter()
init_depth_estimator()
init_pipeline()

DEFAULT_POSITIVE_PROMPT = " commercial product photography"
DEFAULT_NEGATIVE_PROMPT = ""



INTRO = """
#  Image Background Replacement
Minimum recommended hardware: Nvidia A10G large (46 GB RAM, 24 GB VRAM)

## Status
🏝️ Since the publication of this prototype, we've devoted our efforts to developing an enhanced version within Shopify's admin interface, which is now accessible to all Shopify merchants across all subscription plans. This original space is no longer maintained and runs on a CPU-only free tier. Please duplicate this space and utilize your own GPUs.

<hr>

To utilize this tool, first upload the image of your product in either .jpg or .png format. Then, provide a description of the new background you want to replace the original one with. For optimal results, adhere to the following guidelines as shown in the examples below:
1. ❌ Avoid mentioning details about your product in the prompt (e.g., black sneakers)
2. ✅ Do mention how your product is positioned or 'grounded' (e.g., placed on a table)
3. ✅ Do specify the scene you desire (e.g., in a Greek cottage)
4. ✅ Do indicate the style of the image you prefer (e.g., side view commercial product photography)
5. 🤔 Optionally, you can describe what you wish to exclude 🙅 in the negative prompt field.
"""

MORE_INFO = """
### More information
"""


def generate(
    image,
    positive_prompt,
    negative_prompt,
    seed,
    depth_map_feather_threshold,
    depth_map_dilation_iterations,
    depth_map_blur_radius,
    object_caption,
    use_depth_only
):
    if image is None:
        return [None, None, None, None]

    options = {
        'seed': seed,
        'depth_map_feather_threshold': depth_map_feather_threshold,
        'depth_map_dilation_iterations': depth_map_dilation_iterations,
        'depth_map_blur_radius': depth_map_blur_radius,
    }
    print(object_caption)

    return replace_background(image, positive_prompt, negative_prompt, options )


custom_css = """
    #image-upload {
        flex-grow: 1;
    }
    #params .tabs {
        display: flex;
        flex-direction: column;
        flex-grow: 1;
    }
    #params .tabitem[style="display: block;"] {
        flex-grow: 1;
        display: flex !important;
    }
    #params .gap {
        flex-grow: 1;
    }
    #params .form {
        flex-grow: 1 !important;
    }
    #params .form > :last-child{
        flex-grow: 1;
    }
    .md ol, .md ul {
        margin-left: 1rem;
    }
    .md img {
        margin-bottom: 1rem;
    }
"""

with gr.Blocks(css=custom_css) as iface:
    gr.Markdown(INTRO)

    with gr.Row():
        with gr.Column():
            image_upload = gr.Image(
                label="Product image",
                type="pil",
                elem_id="image-upload"
            )
            caption = gr.Label(
                label="Caption",
                visible=developer_mode
            )
        with gr.Column(elem_id="params"):
            with gr.Tab('Prompts'):
                positive_prompt = gr.Textbox(
                    label="Positive Prompt: describe what you'd like to see",
                    lines=3,
                    value=DEFAULT_POSITIVE_PROMPT
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt: describe what you want to avoid",
                    lines=3,
                    value=DEFAULT_NEGATIVE_PROMPT
                )
                caption_ = gr.Textbox(
                    label="Caption: type the caption of the lora model",
                    lines=3
                )
            if developer_mode:
                with gr.Tab('Options'):
                    use_depth_only = gr.Checkbox(label="Use Depth only")
                    seed = gr.Number(
                        label="Seed",
                        precision=0,
                        value=-1,
                        elem_id="seed",
                        visible=developer_mode
                    )
                    depth_map_feather_threshold = gr.Slider(
                        label="Depth map feather threshold",
                        value=128,
                        minimum=0,
                        maximum=255,
                        visible=developer_mode
                    )
                    depth_map_dilation_iterations = gr.Number(
                        label="Depth map dilation iterations",
                        precision=0,
                        value=10,
                        minimum=0,
                        visible=developer_mode
                    )
                    depth_map_blur_radius = gr.Number(
                        label="Depth map blur radius",
                        precision=0,
                        value=10,
                        minimum=0,
                        visible=developer_mode
                    )
            else:
                seed = gr.Number(value=-1, visible=False)
                depth_map_feather_threshold = gr.Slider(
                    value=128, visible=False)
                depth_map_dilation_iterations = gr.Number(
                    precision=0, value=10, visible=False)
                depth_map_blur_radius = gr.Number(
                    precision=0, value=10, visible=False)
                use_depth_only = gr.Checkbox(label="Use Depth only")

    # Enable this button!
    gen_button = gr.Button(
        value="Generate!", variant="primary")

    with gr.Tab('Results'):
        results = gr.Gallery(
            show_label=False,
            object_fit="contain",
            columns=4
        )

    if developer_mode:
        with gr.Tab('Generated'):
            generated = gr.Gallery(
                show_label=False,
                object_fit="contain",
                columns=4
            )

        with gr.Tab('Pre-processing'):
            pre_processing = gr.Gallery(
                show_label=False,
                object_fit="contain",
                columns=4
            )
    else:
        generated = gr.Gallery(visible=False)
        pre_processing = gr.Gallery(visible=False)


    gr.Markdown(MORE_INFO)

    gen_button.click(
        fn=generate,
        inputs=[
            image_upload,
            positive_prompt,
            negative_prompt,
            seed,
            depth_map_feather_threshold,
            depth_map_dilation_iterations,
            depth_map_blur_radius,
            caption_,
            use_depth_only
        ],
        outputs=[
            results,
            generated,
            pre_processing,
            caption
        ],
    )

iface.queue(max_size=10, api_open=False).launch(show_api=False,share=True)
