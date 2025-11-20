# 结合3D影像专家模型的多模态问答
目前多模态模型在2D图像上的理解能力已经取得了极大的进展，但是对于3D影像中类似结节这种微小的病灶还没有成熟的解决方案。尽管如此，类似3D肺结节检测的模型影像模型可以作为专家模型提供给LLM影像信息，LLM给出诊断意见。进一步来说，可以输入任意类型的3D影像，让MLLM决定调用对应的专家模型，然后结合专家模型结果更好的回答问题。
## Demo

## Contents

- [Install](#install)
- [Model Download](#model-download)
- [Serving](#serving)
- [Evaluation](#evaluation)

## Install

1. Clone this repository and navigate to LLaVA-Med folder
```bash
https://github.com/microsoft/LLaVA-Med.git
cd LLaVA-Med
```

2. Install Package: Create conda environment

```Shell
conda create -n llava-med python=3.10 -y
conda activate llava-med
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Model Download


 Model Descriptions | 🤗 Huggingface Hub | 
| --- | ---: |
| LLaVA-Med v1.5 | [microsoft/llava-med-v1.5-mistral-7b](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b) |



## Serving

### Web UI

#### Launch a controller
```Shell
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```

#### Launch a model worker
```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path microsoft/llava-med-v1.5-mistral-7b --multi-modal
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...".

#### Launch a model worker (Multiple GPUs, when GPU VRAM <= 24GB)

If your the VRAM of your GPU is less than 24GB (e.g., RTX 3090, RTX 4090, etc.), you may try running it with multiple GPUs.

```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path microsoft/llava-med-v1.5-mistral-7b --multi-modal --num-gpus 2
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...".


#### Send a test message
```Shell
python -m llava.serve.test_message --model-name llava-med-v1.5-mistral-7b --controller http://localhost:10000
```

#### Launch a gradio web server.
```Shell
python -m llava.serve.gradio_web_server --controller http://localhost:10000
```
#### You can open your browser and chat with a model now.


## Evaluation





## Related Projects

- [LLaVA](https://llava-vl.github.io/)
- [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
- [MMedAgent](https://github.com/Wangyixinxin/MMedAgent)
- [VLM-Radiology-Agent-Framework](https://github.com/Project-MONAI/VLM-Radiology-Agent-Framework)