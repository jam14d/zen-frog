import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Image generation function
def generate_zen_frog(style, pose):
    prompt = f"A {pose.lower()} frog in Japanese {style.lower()} style, minimalist, zen composition, red stamp, off-white background"
    image = pipe(prompt).images[0]
    return image

# Dropdown options
styles = ["Sumi-e (Ink Wash)", "Woodblock", "Watercolor"]
poses = ["Meditating", "Sitting", "Brewing tea", "Floating on a lily pad"]

# Zen haiku description
haiku = (
    "*An old silent pond*\n\n"
    "*A frog jumps into the pond—*\n\n"
    "*Splash! Silence again.*\n\n"
    "— Bashō"
)

# Gradio interface
demo = gr.Interface(
    fn=generate_zen_frog,
    inputs=[
        gr.Dropdown(styles, label="Style"),
        gr.Dropdown(poses, label="Pose")
    ],
    outputs=gr.Image(type="pil", label="Your Zen Frog"),
    title="Zen Frog",
    description=haiku
)

demo.launch()
