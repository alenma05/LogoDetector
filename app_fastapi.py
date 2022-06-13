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
from typing import Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
tags_metadata = [

    {
        "name": "AllianzLogoDetector",
        "description": "This is a project to automatically detect Allianz logo from documents and crop the detected logo in given folder",
        "externalDocs": {
            "url": " http://localhost:5051/",
        },
    },
]

app = FastAPI(title="AllianzLogoDetector",
              description="This is a project to automatically detect Allianz logo from documents and crop the detected logo in given folder",
              version="2.5.0", openapi_tags=tags_metadata)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class LogoDetector(BaseModel):
    input_img_path: str
    pre_trained_model_path: str = None
    cropped_folder: str=None
    bbox_folder: str = r".\bbox"


def plot_results(pil_img, prob, boxes,croped_file_path,model_logo,bbox_file_path):
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    plt.figure(figsize=(16, 10))

    image_array = np.array(pil_img)
    # image_array=np.array(pil_img.getdata()).reshape(pil_img.size[0], pil_img.size[1],3)
    print(image_array.shape)
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    line_width = 5
    list_cropped_logos=[]
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        print(xmin, xmax, ymin, ymax)

        cropped_image = image_array[ceil(ymin) + line_width:ceil(ymax) + line_width,
                        ceil(xmin) + line_width:ceil(xmax) + line_width, :3]

        # plt.imshow(cropped_image\
        cv2.imwrite(croped_file_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        list_cropped_logos.append(croped_file_path)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=5))
        cl = p.argmax()
        # cropped_image = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,fill=False, color=c, linewidth=3)

        text = f'{model_logo.config.id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    #plt.show()
    plt.savefig(bbox_file_path)
    return {"bbox_marked_img":bbox_file_path,"croped_logo_path":list_cropped_logos}

@app.post("/logo/")
def logo_crop_detector(logo:LogoDetector):
    image = Image.open(logo.input_img_path).convert('RGB')
    feature_extractor = AutoFeatureExtractor.from_pretrained(logo.pre_trained_model_path)

    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
    image_array = np.asarray(image)
    #cropped_image = image_array[30:146, 37:81]

    # plt.imshow(cropped_image)
    croped_file_path=f"crop_{Path(logo.input_img_path).name}"
    croped_file_path=os.path.join(logo.cropped_folder,croped_file_path)
    bbox_file_path=f"bbox_{Path(logo.input_img_path).name}"
    bbox_file_path = os.path.join(logo.bbox_folder, bbox_file_path)
    #cv2.imwrite(croped_file_path, cropped_image)


    model_logo = YolosForObjectDetection.from_pretrained(logo.pre_trained_model_path)

    with torch.no_grad():
      outputs = model_logo(pixel_values, output_attentions=True)
    # colors for visualization

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values >= max(probas).tolist()[0]

    # rescale bounding boxes
    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes']


    out=plot_results(image, probas[keep], bboxes_scaled[keep],croped_file_path,model_logo,bbox_file_path)
    return out


if __name__ == "__main__":
    uvicorn.run('app_fastapi:app', host='127.0.0.1', port=5052, debug=True)