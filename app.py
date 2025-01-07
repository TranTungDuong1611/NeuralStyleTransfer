import gradio as gr
import numpy as np
from PIL import Image
from NeuralStyleTransfer import *

style_transfer = NeuralStyleTransfer()

def neural_style_transfer(content_image, style_image, step_size):
    content_image = np.array(content_image)
    style_image = np.array(style_image)
    styled_image = style_transfer(content_image, style_image, step_size)
    
    return styled_image

title = "Neural Style Transfer"
description = "Upload an image and choose a style image to apply the style to the content image."

demo = gr.Interface(
            fn=neural_style_transfer,
            inputs=[
                gr.Image(type="pil", label="Content Image"),
                gr.Image(type="pil", label="Style Image"),
                gr.Slider(minimum=1, maximum=200, step=1, value=1, label="Choose Step Size")
            ],
            outputs="image",
            title=title,
            description=description
        )
demo.launch()