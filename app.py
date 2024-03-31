import io
import glob

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch

from pipeline import Pipeline


MODEL = 'CompVis/stable-diffusion-v1-4'
NORMAL_CKPT = 'ckpt/normal-scene100-notext'
DEPTH_CKPT = 'ckpt/depth-hypersim-notext'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipeline = Pipeline(
        MODEL,
        self_attn_only=False,
        disable_prompts=True,
        onepass=False,
        prediction_type='v_prediction',
        enable_xformers=False,
        device=device,
        mixed_precision='fp16',
)
pipeline.current_demo = ''


def visualize_depth(img):
    """https://matplotlib.org/stable/gallery/user_interfaces/canvasagg.html
    """
    h, w = img.shape
    px = 1/plt.rcParams['figure.dpi']
    fig = Figure(figsize=(w*px, h*px))
    canvas = FigureCanvasAgg(fig)

    ax = fig.subplots()
    ax.imshow(img, cmap='inferno')
    ax.axis('off')
    fig.tight_layout(pad=0)
    ax.margins(0)

    canvas.draw()
    buf = canvas.buffer_rgba()
    arr = np.asarray(buf)
    return arr


def predict_depth(input_img, inference_steps):
    if pipeline.current_demo != 'depth':
        pipeline.define_and_load_lora(DEPTH_CKPT)
        pipeline.current_demo = 'depth'

    trg = pipeline(input_img, inference_steps, 'F')
    trg = (trg - trg.min()) / (trg.max() - trg.min())

    colored = visualize_depth(1-trg)
    np.save('outputs/depth-fp32', trg)
    trg = (trg * 65535).astype(np.uint16)
    Image.fromarray(trg).save('outputs/depth-uint16.png')
    return colored, ['outputs/depth-fp32.npy', 'outputs/depth-uint16.png']


def predict_normal(input_img, inference_steps):
    if pipeline.current_demo != 'normal':
        pipeline.define_and_load_lora(NORMAL_CKPT)
        pipeline.current_demo = 'normal'

    trg = pipeline(input_img, inference_steps, 'RGB')
    return trg


desc = """
- This space is the demo of [Exploiting Diffusion Prior for Generalizable Dense Prediction](https://arxiv.org/abs/2311.18832).
- The model is trained with synthetic images of size 512x512 and pseudo ground truths.
- While the model shows good generalizability, there are sometimes notable errors.
- Please check [our paper](https://arxiv.org/abs/2311.18832) and [Github repo](https://github.com/shinying/dmp) for more details.
"""

normal_demo = gr.Interface(
    fn=predict_normal,
    inputs=[gr.Image(label='Input', type='pil'), 
            gr.Slider(label='Number of generation steps', value=5, minimum=2, maximum=20, step=1)],
    outputs=[gr.Image(label='Output')],
    description=desc,
    allow_flagging='never',
)

depth_demo = gr.Interface(
    fn=predict_depth,
    inputs=[gr.Image(label='Input', type='pil'),
            gr.Slider(label='Number of generation steps', value=5, minimum=2, maximum=20, step=1)],
    outputs=[gr.Image(label='Colored Output'), gr.File(label='Outputs')],
    description=desc,
    allow_flagging='never',
)

demo = gr.TabbedInterface(
    [normal_demo, depth_demo], 
    ['Normal Estimation', 'Depth Prediction'],
    title='DMP Generalizable Dense Prediction',
)
demo.queue(max_size=20)
demo.launch()
