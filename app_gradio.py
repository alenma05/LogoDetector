import matplotlib.pyplot as plt
from math import ceil
from transformers import AutoFeatureExtractor
from transformers import YolosForObjectDetection
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os
import torch
import io
import gradio as gr
import matplotlib.pyplot as plt
import requests, validators
import torch
import pathlib
from PIL import Image
from transformers import AutoFeatureExtractor, DetrForObjectDetection, YolosForObjectDetection
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
cropped_folder=r"./cropped"
pre_trained_path=r"./logo_model_yolos"

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]



def make_prediction(img, feature_extractor, model):
    inputs = feature_extractor(img, return_tensors="pt")
    outputs = model(**inputs)
    img_size = torch.tensor([tuple(reversed(img.size))])
    processed_outputs = feature_extractor.post_process(outputs, img_size)
    return processed_outputs[0]


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def visualize_prediction(pil_img, output_dict, threshold=0.7, id2label=None):
    keep = output_dict["scores"] > threshold
    boxes = output_dict["boxes"][keep].tolist()
    scores = output_dict["scores"][keep].tolist()
    labels = output_dict["labels"][keep].tolist()
    if id2label is not None:
        labels = [id2label[x] for x in labels]

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    line_width=5
    num=0
    list_cropped_path=[]
    for score, (xmin, ymin, xmax, ymax), label, color in zip(scores, boxes, labels, colors):
        num+=1
        croped_file_path = f"crop_{num}_logo.jpg"
        croped_file_path = os.path.join(cropped_folder, croped_file_path)
        image_array=np.array(pil_img)
        cropped_image = image_array[ceil(ymin) + line_width:ceil(ymax) + line_width,
                        ceil(xmin) + line_width:ceil(xmax) + line_width, :3]

        # plt.imshow(cropped_image\
        cv2.imwrite(croped_file_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        list_cropped_path.append(croped_file_path)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=3))
        ax.text(xmin, ymin, f"{label}: {score:0.2f}", fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
        #plt.imshow(cropped_image)
    plt.axis("off")

    # plt.imshow(cropped_image\

    return fig2img(plt.gcf())


def detect_objects(model_name, url_input, image_input, threshold):
    # Extract model and feature extractor
    model_name=model_name
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    #if 'yolos' in model_name:
    model = YolosForObjectDetection.from_pretrained(model_name)
    if validators.url(url_input):
        image = Image.open(requests.get(url_input, stream=True).raw)
    elif image_input:
        image = image_input
    # Make prediction
    processed_outputs = make_prediction(image, feature_extractor, model)
    # Visualize prediction
    viz_img = visualize_prediction(image, processed_outputs, threshold, model.config.id2label)
    return viz_img

def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])
def set_example_url(example: list) -> dict:
    return gr.Textbox.update(value=example[0])

title = """<h1 id="title">Allianz Logo Detection App </h1>"""

description = """
Links to HuggingFace Models:  
- [hustvl/yolos-small](https://huggingface.co/hustvl/yolos-small)
"""

models = [ 'logo_model_yolos']
urls = ["https://c8.alamy.com/comp/J2AB4K/the-new-york-stock-exchange-on-the-wall-street-in-new-york-J2AB4K.jpg"]


css = '''
h1#title {
  text-align: center;
}
'''
demo = gr.Blocks(css=css)

with demo:
    gr.Markdown(title)
    gr.Markdown(description)
    
    options = gr.Dropdown(choices=models, label='Select Object Detection Model', show_label=True)
    slider_input = gr.Slider(minimum=0.2, maximum=1, value=0.95, label='Prediction Threshold')

    with gr.Tabs():
        with gr.TabItem('Image URL'):
            with gr.Row():
                url_input = gr.Textbox(lines=2, label='Enter valid image URL here..')
                img_output_from_url = gr.Image(shape=(650, 650))

            with gr.Row():
                example_url = gr.Dataset(components=[url_input], samples=[[str(url)] for url in urls])

            url_but = gr.Button('Detect')

        with gr.TabItem('Image Upload'):
            with gr.Row():
                img_input = gr.Image(type='pil')
                img_output_from_upload = gr.Image(shape=(650, 650))
                #list_cropped_path=gr.Text()

            with gr.Row():
                example_images = gr.Dataset(components=[img_input],
                                            samples=[[path.as_posix()]
                                                     for path in sorted(pathlib.Path('images_example').rglob('*.JPG'))])

            img_but = gr.Button('Detect')

    url_but.click(detect_objects, inputs=[options, url_input, img_input, slider_input], outputs=img_output_from_url,
                  queue=True)
    img_but.click(detect_objects, inputs=[options, url_input, img_input, slider_input], outputs=img_output_from_upload,
                  queue=True)
    #list_cropped_path=str(list_cropped_path)
    example_images.click(fn=set_example_image, inputs=[example_images], outputs=[img_input])
    example_url.click(fn=set_example_url, inputs=[example_url], outputs=[url_input])


demo.launch(enable_queue=True,share=True)
