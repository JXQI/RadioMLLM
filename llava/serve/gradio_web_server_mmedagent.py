import argparse
import base64
import copy
import datetime
import hashlib
import html
import json
import logging
import os
import re
import tempfile
import time
from collections import defaultdict
from copy import deepcopy
from functools import partial
from glob import glob
from io import BytesIO
from pathlib import Path
from shutil import rmtree
from typing import Dict
from zipfile import ZipFile

import cv2
import gradio as gr
import numpy as np
import pycocotools.mask as mask_util
import requests
import torch
from gradio import processing_utils, utils
from gradio.helpers import Examples
from gradio_client import utils as client_utils
from PIL import Image  # using _ to minimize namespace pollution
from PIL import Image as _Image

from llava.constants import LOGDIR
from llava.conversation import SeparatorStyle, conv_templates, default_conversation
from llava.serve.utils import ImageCache, annotate_xyxy, get_slice_filenames, show_mask
from llava.utils import (
    build_logger,
    moderation_msg,
    server_error_msg,
    violates_moderation,
)

R = partial(round, ndigits=2)


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "MMedAgent Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)


priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}

R = partial(round, ndigits=2)


def b64_encode(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    return img_b64_str


def get_worker_addr(controller_addr, worker_name):
    # get grounding dino addr
    if worker_name.startswith("http"):
        sub_server_addr = worker_name
    else:
        controller_addr = controller_addr
        ret = requests.post(controller_addr + "/refresh_all_workers")
        assert ret.status_code == 200
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        # logger.info(f"Models: {models}")

        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": worker_name}
        )
        sub_server_addr = ret.json()["address"]
    # logger.info(f"worker_name: {worker_name}")
    return sub_server_addr


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown.update(value=model, visible=True)

    state = default_conversation.copy()
    return (
        state,
        dropdown_update,
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
        gr.Gallery.update(visible=True),
    )


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    return (
        state,
        gr.Dropdown.update(choices=models, value=models[0] if len(models) > 0 else ""),
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
        gr.Gallery.update(visible=True),
    )


def change_debug_state(state, with_debug_parameter_from_state, request: gr.Request):
    logger.info(f"change_debug_state. ip: {request.client.host}")
    logger.info("with_debug_parameter_from_state: ", with_debug_parameter_from_state)
    with_debug_parameter_from_state = not with_debug_parameter_from_state

    # modify the text on debug_btn
    debug_btn_value = (
        "Show Progress" if not with_debug_parameter_from_state else "Hide Progress"
    )

    debug_btn_update = gr.Button.update(
        value=debug_btn_value,
    )
    state_update = with_debug_parameter_from_state
    return (
        state,
        state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state),
        "",
        None,
    ) + (debug_btn_update, state_update)


def add_text(
    state,
    text,
    image_dict,
    image_process_mode,
    with_debug_parameter_from_state,
    request: gr.Request,
):
    # dict_keys(['image', 'mask'])
    if image_dict is not None:
        image = image_dict
    else:
        image = None
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (
            state,
            state.to_gradio_chatbot(
                with_debug_parameter=with_debug_parameter_from_state
            ),
            "",
            None,
        ) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (
                state,
                state.to_gradio_chatbot(
                    with_debug_parameter=with_debug_parameter_from_state
                ),
                moderation_msg,
                None,
            ) + (no_change_btn,) * 5

    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if "<image>" not in text:
            text = text + "\n<image>"
        text = (text, image, image_process_mode)
        state = default_conversation.copy()

    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    chatbot_info = state.to_gradio_chatbot(
        with_debug_parameter=with_debug_parameter_from_state
    )
    logger.info(f"chatbot_info: {chatbot_info}")
    logger.info(f"state==={state}")

    return (state, chatbot_info, "", None) + (disable_btn,) * 5
    # return chatbot_info


def http_bot(
    session_state,
    state,
    model_selector,
    temperature,
    top_p,
    max_new_tokens,
    with_debug_parameter_from_state,
    request: gr.Request,
):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (
            state,
            state.to_gradio_chatbot(
                with_debug_parameter=with_debug_parameter_from_state
            ),
        ) + (no_change_btn,) * 6
        return

    if len(state.messages) == state.offset + 2:
        # # First round of conversation
        new_state = conv_templates["mistral_instruct"].copy()
        logger.warning(f"old-state:{state}")
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        # update
        state = new_state
        logger.warning(f"new-state: {state}")

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(
        controller_url + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (
            state,
            state.to_gradio_chatbot(
                with_debug_parameter=with_debug_parameter_from_state
            ),
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
            enable_btn,
        )
        return

    # Construct prompt
    prompt = state.get_prompt()

    # Save images
    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(
            LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg"
        )
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        "model": model_selector,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": (
            state.sep
            if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT]
            else state.sep2
        ),
        # "images": state.get_images(),
    }
    logger.warning(
        f"\n\n==== request ====\n{json.dumps(pload, indent=4, ensure_ascii=False)}\n==== request ====\n\n"
    )
    pload["images"] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (
        state,
        state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state),
    ) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=pload,
            stream=True,
            timeout=10,
        )
        # import ipdb; ipdb.set_trace()
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt) :].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (
                        state,
                        state.to_gradio_chatbot(
                            with_debug_parameter=with_debug_parameter_from_state
                        ),
                    ) + (disable_btn,) * 6
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (
                        state,
                        state.to_gradio_chatbot(
                            with_debug_parameter=with_debug_parameter_from_state
                        ),
                    ) + (
                        disable_btn,
                        disable_btn,
                        disable_btn,
                        enable_btn,
                        enable_btn,
                        enable_btn,
                    )
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        logger.info(f"error: {e}")
        state.messages[-1][-1] = server_error_msg
        yield (
            state,
            state.to_gradio_chatbot(
                with_debug_parameter=with_debug_parameter_from_state
            ),
        ) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn, enable_btn)
        return

    # remove the cursor
    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (
        state,
        state.to_gradio_chatbot(with_debug_parameter=with_debug_parameter_from_state),
    ) + (enable_btn,) * 6

    # check if we need tools
    model_output_text = state.messages[-1][1]
    logger.warning(
        f"model_output_text: {model_output_text}, Now we are going to parse the output."
    )

    # parse the output
    try:
        pattern = r'"thoughts🤔"(.*)"actions🚀"(.*)"value👉"(.*)'
        matches = re.findall(pattern, model_output_text, re.DOTALL)
        # import ipdb; ipdb.set_trace()
        if len(matches) > 0:
            # tool_cfg = json.loads(matches[0][1].strip())
            try:
                tool_cfg = json.loads(matches[0][1].strip())
            except Exception as e:
                tool_cfg = json.loads(matches[0][1].strip().replace("'", '"'))
            logger.info(f"tool_cfg:{tool_cfg}")
        else:
            tool_cfg = None
    except Exception as e:
        logger.error(f"Failed to parse tool config: {e}")
        tool_cfg = None

    # run tool augmentation
    logger.warning(f"trigger tool augmentation with tool_cfg:  {tool_cfg}")
    if tool_cfg is not None and len(tool_cfg) > 0:
        assert (
            len(tool_cfg) == 1
        ), "Only one tool is supported for now, but got: {}".format(tool_cfg)
        api_name = tool_cfg[0]["API_name"]
        logger.info(f"API NAME: {api_name}")
        tool_cfg[0]["API_params"].pop("image", None)
        images = state.get_raw_images()
        if len(images) > 0:
            image = images[0]
        else:
            image = None
        api_paras = {
            "prompt": prompt,
            "box_threshold": 0.3,
            "text_threshold": 0.25,
            "image_id": session_state.image_url.split("/")[-1].split(".")[0],
            "image_root_dir": "/home/tx-deepocean/data1/jxq/code/structured-report/",
            **tool_cfg[0]["API_params"],
        }
        logger.info(f"api_paras:{api_paras}")
        # import ipdb; ipdb.set_trace()
        tool_worker_addr = get_worker_addr(controller_url, api_name)
        logger.info(f"tool_worker_addr: {tool_worker_addr}")
        tool_response = requests.post(
            tool_worker_addr + "/worker_generate",
            headers=headers,
            json=api_paras,
        ).json()

        # build new response
        new_response = f"{api_name} model outputs: {tool_response['lesion_texts']}\n\n"
        first_question = state.messages[-2][-1]
        if isinstance(first_question, tuple):
            first_question = first_question[0].replace("<image>", "")
        first_question = first_question.strip()

        # add new response to the state
        state.append_message(
            state.roles[0],
            new_response
            + "Please summarize the model outputs and answer my first question: {}".format(
                first_question
            ),
        )
        state.append_message(state.roles[1], None)

        # Construct prompt
        prompt2 = state.get_prompt()

        # Make new requests
        pload = {
            "model": model_name,
            "prompt": prompt2,
            "temperature": float(temperature),
            "max_new_tokens": min(int(max_new_tokens), 1536),
            "stop": (
                state.sep
                if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT]
                else state.sep2
            ),
            "images": f"List of {len(state.get_images())} images: {all_image_hash}",
        }
        logger.warning(
            f"\n\n==== request ====\n{json.dumps(pload, indent=4, ensure_ascii=False)}\n==== request ==== \n\n"
        )
        pload["images"] = state.get_images()

        state.messages[-1][-1] = "▌"
        yield (
            state,
            state.to_gradio_chatbot(
                with_debug_parameter=with_debug_parameter_from_state
            ),
        ) + (disable_btn,) * 6

        try:
            # Stream output
            response = requests.post(
                worker_addr + "/worker_generate_stream",
                headers=headers,
                json=pload,
                stream=True,
                timeout=10,
            )
            # import ipdb; ipdb.set_trace()
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode())
                    if data["error_code"] == 0:
                        output = data["text"][len(prompt2) :].strip()
                        # TODO
                        output = output.replace("<", "(")
                        output = output.replace(">", ")")
                        state.messages[-1][-1] = output + "▌"
                        yield (
                            state,
                            state.to_gradio_chatbot(
                                with_debug_parameter=with_debug_parameter_from_state
                            ),
                        ) + (disable_btn,) * 6
                    else:
                        output = data["text"] + f" (error_code: {data['error_code']})"
                        state.messages[-1][-1] = output
                        yield (
                            state,
                            state.to_gradio_chatbot(
                                with_debug_parameter=with_debug_parameter_from_state
                            ),
                        ) + (
                            disable_btn,
                            disable_btn,
                            disable_btn,
                            enable_btn,
                            enable_btn,
                            enable_btn,
                        )
                        return
                    time.sleep(0.03)
        except requests.exceptions.RequestException as e:
            state.messages[-1][-1] = server_error_msg
            yield (
                state,
                state.to_gradio_chatbot(
                    with_debug_parameter=with_debug_parameter_from_state
                ),
            ) + (
                disable_btn,
                disable_btn,
                disable_btn,
                enable_btn,
                enable_btn,
                enable_btn,
            )
            return

        # remove the cursor
        state.messages[-1][-1] = state.messages[-1][-1][:-1]

        def get_lesion_pairs(output: str, lesion_slices: Dict):

            from PIL import Image

            logger.warning(state.messages[-1])
            pairs = []
            for _lesion_name in re.findall(r"Img\d+", output):
                _lesion_img = Image.open(lesion_slices[_lesion_name])
                # _lesion_img = _lesion_img.resize((128, 128))
                pairs.append((_lesion_img, _lesion_name))
            logger.warning(pairs)
            return pairs

        lesion_pairs = get_lesion_pairs(
            state.messages[-1][-1], tool_response["lesion_slices"]
        )
        yield (
            state,
            state.to_gradio_chatbot(
                with_debug_parameter=with_debug_parameter_from_state
            ),
            lesion_pairs,
        ) + (enable_btn,) * 6

    finish_tstamp = time.time()

    # FIXME: disabled temporarily for image generation.
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "state": state.dict(force_str=True),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


IMG_URLS_OR_PATHS = {
    "CE571001-1965456-30423-4": "/home/tx-deepocean/data1/jxq/code/structured-report/data/niigz/CE571001-1965456-30423-4.nii.gz",
    "CE571001-1768995-29744-4": "/home/tx-deepocean/data1/jxq/code/structured-report/data/niigz/CE571001-1768995-29744-4.nii.gz",
    "CE571001-1951147-13386582-4": "/home/tx-deepocean/data1/jxq/code/structured-report/data/niigz/CE571001-1951147-13386582-4.nii.gz",
    "CE571001-1771535-30158-4": "/home/tx-deepocean/data1/jxq/code/structured-report/data/niigz/CE571001-1771535-30158-4.nii.gz",
}
CACHED_IMAGES = ImageCache(
    cache_dir="/home/tx-deepocean/data1/jxq/code/VLM-Radiology-Agent-Framework/m3/demo/cache_images/"
)

EXAMPLE_PROMPTS_3D = [
    ["该图像中总共有几个结节？"],
    ["找出该图像中结节体积最大的结节？"],
    ["找出该图中恶性风险最高的结节．"],
    ["根据该图诊断结果给出诊断意见"],
    ["该图中恶性结节有哪些？"],
    ["找出该图像中结节体积最大的结节？"],
    ["图中左肺的结节有哪些？"],
]


class SessionVariables:
    """Class to store the session variables"""

    def __init__(self):
        """Initialize the session variables"""
        self.slice_index = None  # Slice index for 3D images
        self.image_url = None  # Image URL to the image on the web
        self.backup = (
            {}
        )  # Cached varaiables from previous messages for the current conversation
        self.axis = 2
        self.top_p = 0.9
        self.temperature = 0.0
        self.max_tokens = 1024
        self.temp_working_dir = None
        self.idx_range = (None, None)
        self.interactive = False
        self.sys_msgs_to_hide = []
        self.modality_prompt = "Auto"
        self.img_urls_or_paths = IMG_URLS_OR_PATHS

    def restore_from_backup(self, attr):
        """Retrieve the attribute from the backup"""
        attr_val = self.backup.get(attr, None)
        if attr_val is not None:
            self.__setattr__(attr, attr_val)


def input_image(image, state: SessionVariables):
    """Update the session variables with the input image data URL if it's inputted by the user"""
    logger.debug(f"Received user input image")
    # TODO: support user uploaded images
    return image, state


def update_image_selection(selected_image, state: SessionVariables, slice_index=None):
    """Update the gradio components based on the selected image"""
    logger.debug(f"Updating display image for {selected_image}")
    state.image_url = state.img_urls_or_paths.get(selected_image, None)
    img_file = CACHED_IMAGES.get(state.image_url, None)
    logger.info(f"====={state.image_url}****{img_file}=========")

    if state.image_url is None or img_file is None:
        return None, state, gr.Slider(0, 2, 1, step=1, visible=False), [[""]]

    state.interactive = True
    if img_file.endswith(".nii.gz"):
        if slice_index is None:
            slice_file_pttn = img_file.replace(".nii.gz", "_slice*_img.jpg")
            # glob the image files
            slice_files = glob(slice_file_pttn)
            state.slice_index = len(slice_files) // 2
            state.idx_range = (0, len(slice_files) - 1)
        else:
            # Slice index is updated by the slidebar.
            # There is no need to update the idx_range.
            state.slice_index = slice_index

        image_filename = get_slice_filenames(img_file, state.slice_index)
        if not os.path.exists(os.path.join(CACHED_IMAGES.dir(), image_filename)):
            raise ValueError(f"Image file {image_filename} does not exist.")

        return (
            os.path.join(CACHED_IMAGES.dir(), image_filename),
            state,
            gr.Slider.update(
                minimum=state.idx_range[0],
                maximum=state.idx_range[1],
                value=state.slice_index,
                step=1,
                visible=True,
                interactive=True,
            ),
        )

    state.slice_index = None
    state.idx_range = (None, None)
    return (
        img_file,
        state,
        gr.Slider(0, 2, 1, 0, visible=False),
        gr.Dataset.update(samples=EXAMPLE_PROMPTS_2D),
    )


def update_temperature(temperature, state):
    """Update the temperature"""
    logger.debug(f"Updating the temperature")
    state.temperature = temperature
    return state


def update_top_p(top_p, state):
    """Update the top P"""
    logger.debug(f"Updating the top P")
    state.top_p = top_p
    return state


def update_max_tokens(max_tokens, state):
    """Update the max tokens"""
    logger.debug(f"Updating the max tokens")
    state.max_tokens = max_tokens
    return state


title_markdown = """
# 👨‍⚕️👩‍⚕️ MMedAgent: Learning to Use Medical Tools with Multi-modal Agent
[[Paper]](https://arxiv.org/abs/2407.02483) [[Code]](https://github.com/Wangyixinxin/MMedAgent)
"""

tos_markdown = """
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
"""


learn_more_markdown = """
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
"""


def build_demo(embed_mode):
    textbox = gr.Textbox(
        show_label=False,
        placeholder="Enter text and press ENTER",
        visible=True,
        container=False,
    )
    with gr.Blocks(title="MMedAgent", theme=gr.themes.Base()) as demo:
        session_state = gr.State(value=SessionVariables())
        conv_state = gr.State(value=default_conversation.copy())

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False,
                    )

                # imagebox = ImageMask()
                imagebox = gr.Image(label="Image", type="pil")
                image_dropdown = gr.Dropdown(
                    label="Select an image",
                    choices=["Please select .."]
                    + list(session_state.value.img_urls_or_paths.keys()),
                )
                image_slider = gr.Slider(0, 2, 1, step=0, visible=False)
                with gr.Accordion("Parameters", open=False) as parameter_row:
                    image_process_mode = gr.Radio(
                        ["Crop", "Resize", "Pad"],
                        value="Crop",
                        label="Preprocess for non-square image",
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=0,
                        maximum=2048,
                        value=512,
                        step=64,
                        interactive=True,
                        label="Max output tokens",
                    )
                    # with_debug_parameter_check_box = gr.Checkbox(label="With debug parameter", checked=args.with_debug_parameter)

            with gr.Column(scale=6):
                chatbot = gr.Chatbot(
                    elem_id="chatbot", label="MMedAgent Chatbot", height=400
                )
                lesionbox = gr.Gallery(
                    label="lesion", height=100, columns=10, visible=True
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=60):
                        submit_btn = gr.Button(value="Submit")
                with gr.Row(visible=True) as button_row:
                    debug_btn = gr.Button(value="Show Progress", interactive=True)
                    # import ipdb; ipdb.set_trace()
                if args.with_debug_parameter:
                    debug_btn.value = "Show Progress"
                with_debug_parameter_state = gr.State(
                    value=args.with_debug_parameter,
                )

        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)
        imagebox.change(
            fn=input_image,
            inputs=[imagebox, session_state],
            outputs=[imagebox, session_state],
        )
        image_dropdown.change(
            fn=update_image_selection,
            inputs=[image_dropdown, session_state],
            outputs=[imagebox, session_state, image_slider],
        )
        image_slider.release(
            fn=update_image_selection,
            inputs=[image_dropdown, session_state, image_slider],
            outputs=[imagebox, session_state, image_slider],
        )
        temperature_slider.change(
            fn=update_temperature,
            inputs=[temperature_slider, session_state],
            outputs=[session_state],
        )
        top_p_slider.change(
            fn=update_top_p,
            inputs=[top_p_slider, session_state],
            outputs=[session_state],
        )
        max_tokens_slider.change(
            fn=update_max_tokens,
            inputs=[max_tokens_slider, session_state],
            outputs=[session_state],
        )

        # Register listeners

        textbox.submit(
            add_text,
            [
                conv_state,
                textbox,
                imagebox,
                image_process_mode,
                with_debug_parameter_state,
            ],
            [conv_state, chatbot, textbox, imagebox, debug_btn],
        ).then(
            http_bot,
            [
                session_state,
                conv_state,
                model_selector,
                temperature_slider,
                top_p_slider,
                max_tokens_slider,
                with_debug_parameter_state,
            ],
            [conv_state, chatbot, lesionbox, debug_btn],
        )

        submit_btn.click(
            add_text,
            [
                conv_state,
                textbox,
                imagebox,
                image_process_mode,
                with_debug_parameter_state,
            ],
            [conv_state, chatbot, textbox, imagebox, debug_btn],
        ).then(
            http_bot,
            [
                session_state,
                conv_state,
                model_selector,
                temperature_slider,
                top_p_slider,
                max_tokens_slider,
                with_debug_parameter_state,
            ],
            [conv_state, chatbot, lesionbox, debug_btn],
        )

        debug_btn.click(
            change_debug_state,
            [conv_state, with_debug_parameter_state],
            [conv_state, chatbot, textbox, imagebox]
            + [debug_btn, with_debug_parameter_state],
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [
                    conv_state,
                    model_selector,
                    chatbot,
                    textbox,
                    submit_btn,
                    button_row,
                    parameter_row,
                ],
                _js=get_window_url_params,
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [
                    conv_state,
                    model_selector,
                    chatbot,
                    textbox,
                    submit_btn,
                    button_row,
                    parameter_row,
                ],
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--controller-url", type=str, default="http://localhost:20001")
    parser.add_argument("--concurrency-count", type=int, default=8)
    parser.add_argument(
        "--model-list-mode", type=str, default="once", choices=["once", "reload"]
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--with_debug_parameter", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()
    models = [i for i in models if "llava" in i]

    logger.info(args)
    CACHED_IMAGES.cache(IMG_URLS_OR_PATHS)
    demo = build_demo(args.embed)
    _app, local_url, share_url = demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=True
    ).launch(
        server_name=args.host, server_port=args.port, share=args.share, debug=args.debug
    )
    logger.info("Local URL: ", local_url)
    logger.info("Share URL: ", share_url)
