from fastrtc import Stream
import gradio as gr
import cv2
from huggingface_hub import hf_hub_download
from .inference import YOLOv10

model_file = hf_hub_download(
    repo_id="onnx-community/yolov10n", filename="onnx/model.onnx"
)

# git clone https://huggingface.co/spaces/fastrtc/object-detection
# for YOLOv10 implementation
model = YOLOv10(model_file)

def detection(image, conf_threshold=0.3):
    image = cv2.resize(image, (model.input_width, model.input_height))
    new_image = model.detect_objects(image, conf_threshold)
    return cv2.resize(new_image, (500, 500))

stream = Stream(
    handler=detection,
    modality="video", 
    mode="send-receive",
    additional_inputs=[
        gr.Slider(minimum=0, maximum=1, step=0.01, value=0.3)
    ]
)
stream.ui.launch()
