from transformers import pipeline

captioner = None
PROMPT = "The main subject of this picture is a"


def init():
    global captioner




def derive_caption(image):
    print("Initializing captioner...")

    captioner = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
        prompt=PROMPT
    )
    result = captioner(image, max_new_tokens=20)
    raw_caption = result[0]["generated_text"]
    caption = raw_caption.lower().replace(PROMPT.lower(), "").strip()
    return caption
