import argparse
import json
import os
import re
import time
from glob import glob
from typing import Dict, List, Tuple

import gradio as gr
import requests

from llava.conversation import SeparatorStyle, default_conversation
from llava.serve.utils import ImageCache, get_slice_filenames
from llava.utils import (
    build_logger,
    server_error_msg,
)

##### global settings #####
logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "MMedAgent Client"}

IMG_URLS_OR_PATHS = {
    # CHEST
    "CE571001-1965456-30423-4": "/home/tx-deepocean/data2/jxq/data/mmedagent/processed/demo/chest/niigz/CE571001-1965456-30423-4.nii.gz",
    "CE571001-1768995-29744-4": "/home/tx-deepocean/data2/jxq/data/mmedagent/processed/demo/chest/niigz/CE571001-1768995-29744-4.nii.gz",
    "CE571001-1951147-13386582-4": "/home/tx-deepocean/data2/jxq/data/mmedagent/processed/demo/chest/niigz/CE571001-1951147-13386582-4.nii.gz",
    "CE571001-1771535-30158-4": "/home/tx-deepocean/data2/jxq/data/mmedagent/processed/demo/chest/niigz/CE571001-1771535-30158-4.nii.gz",
    # HEART
    "CN010021-24474-7": "/home/tx-deepocean/data2/jxq/data/mmedagent/processed/demo/heart/val_niigz/CN010021-24474-7.nii.gz",
    "CN010023-1811071134-2765-501": "/home/tx-deepocean/data2/jxq/data/mmedagent/processed/demo/heart/val_niigz/CN010023-1811071134-2765-501.nii.gz",
    "CN411002-3683977-R02439383": "/home/tx-deepocean/data2/jxq/data/mmedagent/processed/demo/heart/val_niigz/CN411002-3683977-R02439383.nii.gz",
    "CN533002-HDH358703-5724-601": "/home/tx-deepocean/data2/jxq/data/mmedagent/processed/demo/heart/val_niigz/CN533002-HDH358703-5724-601.nii.gz",
}
CACHED_IMAGES = ImageCache(
    cache_dir="/home/tx-deepocean/data2/jxq/data/mmedagent/processed/demo/cache_images"
)
##############################


class SessionVariables:
    """Class to store the session variables"""

    def __init__(self, with_debug_parameter=False):
        """Initialize the session variables"""
        self.slice_index = None  # Slice index for 3D images
        self.image_url = None  # Image URL to the image on the web
        self.backup = (
            {}
        )  # Cached varaiables from previous messages for the current conversation
        self.axis = 2
        self.top_p = 0.7
        self.temperature = 0.2
        self.max_tokens = 512
        self.temp_working_dir = None
        self.idx_range = (None, None)
        self.interactive = False
        self.sys_msgs_to_hide = []
        self.modality_prompt = "Auto"
        self.img_urls_or_paths = IMG_URLS_OR_PATHS

        self.with_debug_parameter = with_debug_parameter

    def restore_from_backup(self, attr):
        """Retrieve the attribute from the backup"""
        attr_val = self.backup.get(attr, None)
        if attr_val is not None:
            self.__setattr__(attr, attr_val)

    @staticmethod
    def get_worker_addr(controller_addr, worker_name):
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
        return sub_server_addr


class Interactions:

    @staticmethod
    def update_image_selection(
        image_dropdown: gr.Dropdown,
        param_state: SessionVariables,
        image_slider: gr.Slider = None,
    ) -> Tuple[gr.Image, SessionVariables, gr.Slider]:

        param_state.image_url = param_state.img_urls_or_paths.get(image_dropdown, None)
        img_file = CACHED_IMAGES.get(param_state.image_url, None)

        if param_state.image_url is None or img_file is None:
            _debug_info = json.dumps(
                {"img_file": img_file, "slice_index": image_slider},
                indent=4,
                ensure_ascii=False,
            )
            logger.debug(f"update_image_selection: {_debug_info}")
            return None, param_state, gr.Slider(0, 2, 1, step=1, visible=False)

        if img_file.endswith(".nii.gz"):
            if image_slider is None:
                slice_file_pttn = img_file.replace(".nii.gz", "_slice*_img.jpg")
                # glob the image files
                slice_files = glob(slice_file_pttn)
                param_state.slice_index = len(slice_files) // 2
                param_state.idx_range = (0, len(slice_files) - 1)
            else:
                # Slice index is updated by the slidebar.
                # There is no need to update the idx_range.
                param_state.slice_index = image_slider

            image_filename = get_slice_filenames(img_file, param_state.slice_index)
            if not os.path.exists(os.path.join(CACHED_IMAGES.dir(), image_filename)):
                raise ValueError(f"Image file {image_filename} does not exist.")

            _debug_info = json.dumps(
                {"img_file": img_file, "slice_index": param_state.slice_index},
                indent=4,
                ensure_ascii=False,
            )
            logger.debug(f"update_image_selection: {_debug_info}")
            return (
                os.path.join(CACHED_IMAGES.dir(), image_filename),
                param_state,
                gr.Slider.update(
                    minimum=param_state.idx_range[0],
                    maximum=param_state.idx_range[1],
                    value=param_state.slice_index,
                    step=1,
                    visible=True,
                    interactive=True,
                ),
            )
        else:
            param_state.slice_index = None
            param_state.idx_range = (None, None)
            _debug_info = json.dumps(
                {"img_file": img_file, "slice_index": 1}, indent=4, ensure_ascii=False
            )
            logger.debug(f"update_image_selection: {_debug_info}")
            return (
                img_file,
                param_state,
                gr.Slider(0, 2, 1, 0, visible=True),
            )

    @staticmethod
    def update_temperature(
        temperature: gr.Slider, param_state: SessionVariables
    ) -> SessionVariables:
        """Update the temperature"""
        logger.debug(f"Updating the temperature: {temperature}")
        param_state.temperature = temperature
        return param_state

    @staticmethod
    def update_top_p(top_p: gr.Slider, param_state: SessionVariables):
        """Update the top P"""
        logger.debug(f"Updating the top P: {top_p}")
        param_state.top_p = top_p
        return param_state

    @staticmethod
    def update_max_tokens(max_tokens: gr.Slider, param_state: SessionVariables):
        """Update the max tokens"""
        logger.debug(f"Updating the max tokens")
        param_state.max_tokens = max_tokens
        return param_state

    @staticmethod
    def add_text(
        param_state: gr.State,
        conv_state: gr.State,
        text: gr.Textbox,
        image: gr.Image,
        image_process_mode: gr.Radio,
        request: gr.Request,
    ) -> Tuple[gr.State, gr.State, gr.Chatbot, gr.Textbox]:

        if len(text) <= 0 and image is None:
            conv_state.skip_next = True
            logger.warning("add_text: text or image is empty!")

            return (
                param_state,
                conv_state,
                conv_state.to_gradio_chatbot(param_state.with_debug_parameter),
            )

        text = text[:1536]  # TODO：没想通为什么？
        if image is not None:
            text = text[:1200]  # TODO:这些参数哪里来的？
            if "<image>" not in text:
                text = text + "\n<image>"
            text = (text, image, image_process_mode)
            conv_state = default_conversation.copy()

        conv_state.append_message(conv_state.roles[0], text)
        conv_state.append_message(conv_state.roles[1], None)
        conv_state.skip_next = False

        logger.debug(f"add_text: text:{text}")
        return (
            param_state,
            conv_state,
            conv_state.to_gradio_chatbot(param_state.with_debug_parameter),
            None,
        )

    @staticmethod
    def request(
        param_state: gr.State,
        conv_state: gr.State,
        worker_addr: str,
        pload: str,
        tool_response: Dict,
    ) -> Tuple[gr.State, gr.Chatbot, gr.Gallery]:

        try:
            # Stream output
            logger.warning(worker_addr)
            response = requests.post(
                worker_addr + "/worker_generate_stream",
                headers=headers,
                json=pload,
                stream=True,
                timeout=10,
            )
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode())
                    if data["error_code"] == 0:
                        output = data["text"][
                            len(pload["prompt"]) :
                        ].strip()  # TODO:为什么要截取？
                        output = output.replace("<", "(")
                        output = output.replace(">", ")")
                        conv_state.messages[-1][-1] = output + "▌"
                        yield (
                            conv_state,
                            conv_state.to_gradio_chatbot(
                                param_state.with_debug_parameter
                            ),
                            Interactions.get_lesion_pairs(
                                conv_state.messages[-1][-1], tool_response
                            ),
                        )
                    else:
                        output = data["text"] + f" (error_code: {data['error_code']})"
                        conv_state.messages[-1][-1] = output
                        yield (
                            conv_state,
                            conv_state.to_gradio_chatbot(
                                param_state.with_debug_parameter
                            ),
                            None,
                        )
                    time.sleep(0.03)
        except requests.exceptions.RequestException as e:
            logger.info(f"error: {e}")
            conv_state.messages[-1][-1] = server_error_msg
            yield (
                conv_state,
                conv_state.to_gradio_chatbot(param_state.with_debug_parameter),
                None,
            )
        # finally:
        # if "lesion_slices" in tool_response:
        #     lesion_pairs = Interactions.get_lesion_pairs(conv_state.messages[-1][-1], tool_response['lesion_slices'])
        # else:
        #     lesion_pairs = []
        # yield (
        #     conv_state,
        #     conv_state.to_gradio_chatbot(param_state.with_debug_parameter),
        #     lesion_pairs,
        # )

    @staticmethod
    def http_bot(
        param_state: gr.State,
        conv_state: gr.State,
        model_selector: gr.Dropdown,
        request: gr.Request,
    ) -> Tuple[gr.State, gr.Chatbot, gr.Gallery]:

        logger.info(f"http_bot. ip: {request.client.host}")

        # Query worker address
        controller_url = args.controller_url
        ret = requests.post(
            controller_url + "/get_worker_address", json={"model": model_selector}
        )
        worker_addr = ret.json()["address"]
        logger.info(
            json.dumps(
                {"model_name": model_selector, "worker_addr": worker_addr},
                indent=4,
                ensure_ascii=False,
            )
        )
        if worker_addr == "":
            logger.error("http_bot: No available worker")
            conv_state.messages[-1][-1] = server_error_msg
            yield (
                conv_state,
                conv_state.to_gradio_chatbot(param_state.with_debug_parameter),
                None,
            )
            return

        # Construct promt
        if conv_state.skip_next:
            logger.warning(
                "http_bot: This generate call is skipped due to invalid inputs"
            )
            yield (
                conv_state,
                conv_state.to_gradio_chatbot(param_state.with_debug_parameter),
                None,
            )
            return
        prompt = conv_state.get_prompt()

        # Make requests
        pload = {
            "model": model_selector,
            "prompt": prompt,
            "temperature": float(param_state.temperature),
            "top_p": float(param_state.top_p),
            "max_new_tokens": min(int(param_state.max_tokens), 1536),
            "stop": (
                conv_state.sep
                if conv_state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT]
                else conv_state.sep2
            ),
            "images": conv_state.get_images(),
        }
        logger.debug(
            f"\n\n==== request ====\n{json.dumps(pload, indent=4, ensure_ascii=False)}\n==== request ====\n\n"
        )
        conv_state.messages[-1][-1] = "▌"
        yield (
            conv_state,
            conv_state.to_gradio_chatbot(param_state.with_debug_parameter),
            None,
        )

        for conv_state, chat_bot, lesion_bot in Interactions.request(
            param_state, conv_state, worker_addr, pload, {}
        ):
            yield (conv_state, chat_bot, lesion_bot)

        # remove the cursor
        conv_state.messages[-1][-1] = conv_state.messages[-1][-1][:-1]
        yield (
            conv_state,
            conv_state.to_gradio_chatbot(param_state.with_debug_parameter),
            None,
        )

        # check if we need tools
        model_output_text = conv_state.messages[-1][1]
        logger.warning(
            f"model_output_text: {model_output_text}, Now we are going to parse the output."
        )

        # step2: parse the output
        try:
            pattern = r'"thoughts🤔"(.*)"actions🚀"(.*)"value👉"(.*)'
            matches = re.findall(pattern, model_output_text, re.DOTALL)
            if len(matches) > 0:
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
            tool_cfg[0]["API_params"].pop("image", None)
            api_paras = {
                "prompt": prompt,
                "api_name": api_name,
                "image_id": param_state.image_url.split("/")[-1].split(".")[0],
                **tool_cfg[0]["API_params"],
            }
            logger.debug(
                f"api_paras:{json.dumps(api_paras, indent=4, ensure_ascii=False)}"
            )

            tool_worker_addr = SessionVariables.get_worker_addr(controller_url, "CHEST")
            logger.info(f"tool_worker_addr: {tool_worker_addr}")
            tool_response = requests.post(
                tool_worker_addr + "/worker_generate",
                headers=headers,
                json=api_paras,
            ).json()

            # step3: build new response
            new_response = (
                f"{api_name} model outputs: {tool_response['lesion_texts']}\n\n"
            )
            first_question = conv_state.messages[-2][-1]
            if isinstance(first_question, tuple):
                first_question = first_question[0].replace("<image>", "")
            first_question = first_question.strip()

            # add new response to the state
            conv_state.append_message(
                conv_state.roles[0],
                new_response
                + "Please summarize the model outputs and answer my first question: {}".format(
                    first_question
                ),
            )
            conv_state.append_message(conv_state.roles[1], None)

            # Construct prompt
            prompt2 = conv_state.get_prompt()
            # Make new requests
            pload = {
                "model": model_selector,
                "prompt": prompt2,
                "temperature": float(param_state.temperature),
                "top_p": float(param_state.top_p),
                "max_new_tokens": min(int(param_state.max_tokens), 1536),
                "stop": (
                    conv_state.sep
                    if conv_state.sep_style
                    in [SeparatorStyle.SINGLE, SeparatorStyle.MPT]
                    else conv_state.sep2
                ),
                "images": conv_state.get_images(),
            }
            logger.info(
                f"\n\n==== request ====\n{json.dumps(pload, indent=4, ensure_ascii=False)}\n==== request ==== \n\n"
            )
            conv_state.messages[-1][-1] = "▌"
            yield (
                conv_state,
                conv_state.to_gradio_chatbot(param_state.with_debug_parameter),
                None,
            )

            # get output from vllm
            for conv_state, chat_bot, lesion_bot in Interactions.request(
                param_state, conv_state, worker_addr, pload, tool_response
            ):
                yield (conv_state, chat_bot, lesion_bot)

            # remove the cursor
            conv_state.messages[-1][-1] = conv_state.messages[-1][-1][:-1]

            yield (
                conv_state,
                conv_state.to_gradio_chatbot(param_state.with_debug_parameter),
                lesion_bot,
            )

    @staticmethod
    def get_lesion_pairs(output: str, tool_response: Dict) -> List[Tuple[str, str]]:
        pairs = []
        if "lesion_slices" in tool_response:
            _lesion_slices = tool_response["lesion_slices"]
            # pattern1 = r'Img\d+_vessel[\w-]+'  # 匹配 Img0_vessel2-p-1 格式
            # pattern2 = r'Img\d+'               # 匹配 Img12 格式
            pattern = r"Img\d+_vessel[\w-]+|Img\d+(?=\s|$|,|\))"
            lesion_matches = re.findall(pattern, output)
            print(lesion_matches, "=========")
            for _lesion_name in lesion_matches:
                print(_lesion_name, "----")
                if "vessel" in _lesion_name:
                    _lesion_name = _lesion_name.split("_")[-1]
                print(_lesion_name, "+++++")
                pairs.append((_lesion_slices[_lesion_name], _lesion_name))

        return pairs

    @staticmethod
    def change_debug_state(
        param_state: gr.State, conv_state: gr.State, request: gr.Request
    ) -> Tuple[gr.Chatbot, gr.Button]:
        param_state.with_debug_parameter = not param_state.with_debug_parameter
        debug_btn_value = (
            "Show Progress" if not param_state.with_debug_parameter else "Hide Progress"
        )

        debug_btn_update = gr.Button.update(
            value=debug_btn_value,
        )
        return (
            conv_state.to_gradio_chatbot(
                with_debug_parameter=param_state.with_debug_parameter
            ),
            debug_btn_update,
        )


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    logger.info(f"Models: {models}")
    return models


def build_demo(args):
    with gr.Blocks(title="MMedAgent", theme=gr.themes.Base()) as demo:
        param_state = gr.State(
            value=SessionVariables(with_debug_parameter=args.with_debug_parameter)
        )
        conv_state = gr.State(value=default_conversation.copy())

        with gr.Row():
            # 图像选择&&参数设置
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False,
                    )
                image_box = gr.Image(label="Image", type="pil")
                image_dropdown = gr.Dropdown(
                    label="Select an image",
                    choices=["Please select .."]
                    + list(param_state.value.img_urls_or_paths.keys()),
                    max_choices=5,  # 限制显示数量
                    filterable=True,  # 启用搜索
                    allow_custom_value=False,  # 禁止自定义输入
                    multiselect=False,  # 单选模式
                    # 布局控制
                    container=True,
                    min_width=350,
                    scale=1,
                    # 用户体验
                    interactive=True,
                    info="输入图像名称进行搜索",
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
                        value=param_state.value.temperature,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    topP_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=param_state.value.top_p,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                    )
                    maxTokens_slider = gr.Slider(
                        minimum=0,
                        maximum=2048,
                        value=param_state.value.max_tokens,
                        step=64,
                        interactive=True,
                        label="Max output tokens",
                    )
            # 交互部分
            with gr.Column(scale=6):
                chat_bot = gr.Chatbot(
                    elem_id="chatbot", label="MMedAgent Chatbot", height=500
                )
                lesion_box = gr.Gallery(
                    label="lesion", height=100, columns=10, visible=True
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        text_box = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press ENTER",
                            visible=True,
                            container=False,
                        )
                    with gr.Column(scale=1, min_width=60):
                        submit_btn = gr.Button(value="Submit")
                with gr.Row(visible=True) as button_row:
                    debug_btn = gr.Button(value="Show Progress", interactive=True)
                    # import ipdb; ipdb.set_trace()
                if param_state.value.with_debug_parameter:
                    debug_btn.value = "Show Progress"

        # 交互操作-修改输入参数
        image_dropdown.change(
            fn=Interactions.update_image_selection,
            inputs=[image_dropdown, param_state],
            outputs=[image_box, param_state, image_slider],
        )
        image_slider.release(
            fn=Interactions.update_image_selection,
            inputs=[image_dropdown, param_state, image_slider],
            outputs=[image_box, param_state, image_slider],
        )
        temperature_slider.change(
            fn=Interactions.update_temperature,
            inputs=[temperature_slider, param_state],
            outputs=[param_state],
        )
        topP_slider.change(
            fn=Interactions.update_top_p,
            inputs=[topP_slider, param_state],
            outputs=[param_state],
        )
        maxTokens_slider.change(
            fn=Interactions.update_max_tokens,
            inputs=[maxTokens_slider, param_state],
            outputs=[param_state],
        )

        # 交互操作－提问
        text_box.submit(
            Interactions.add_text,
            [param_state, conv_state, text_box, image_box, image_process_mode],
            [param_state, conv_state, chat_bot, text_box],
        ).then(
            Interactions.http_bot,
            [
                param_state,
                conv_state,
                model_selector,
            ],
            [conv_state, chat_bot, lesion_box],
        )

        submit_btn.click(
            Interactions.add_text,
            [param_state, conv_state, text_box, image_box, image_process_mode],
            [conv_state, conv_state, chat_bot, text_box],
        ).then(
            Interactions.http_bot,
            [param_state, conv_state, model_selector],
            [conv_state, chat_bot, lesion_box],
        )

        debug_btn.click(
            Interactions.change_debug_state,
            [param_state, conv_state],
            [chat_bot, debug_btn],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--controller-url", type=str, default="http://localhost:20001")
    parser.add_argument("--concurrency-count", type=int, default=8)
    parser.add_argument(
        "--model_list_mode", type=str, default="once", choices=["once", "reload"]
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
    demo = build_demo(args)
    _app, local_url, share_url = demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=True
    ).launch(
        server_name=args.host, server_port=args.port, share=args.share, debug=args.debug
    )
    logger.info("Local URL: ", local_url)
    logger.info("Share URL: ", share_url)
