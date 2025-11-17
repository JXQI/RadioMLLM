import argparse
import copy
import json
import os
import random

from tqdm import tqdm

thoughts_options = [
    "这是一张2D图像，不涉及复杂的医学分析，我将直接基于视觉特征提供回答。",
    "由于这是2D图像，无需调用专家模型，我能够直接分析并给出答案。",
    "这是一张二维图像，专家模型的优势在于3D分析，因此我将直接解读图像内容。",
    "针对这张2D图像，我可以直接进行视觉分析，无需借助专家模型。",
    "这是一张简单的2D图像，不涉及深度医学计算，我将直接提供观察结果。",
    "由于图像为2D格式，我能够直接处理并回答相关问题，无需专家介入。",
    "这是一张二维切片图像，专家模型无法提供额外价值，我将直接分析。",
    "对于这张2D图像，我可以基于基础视觉能力直接给出准确回答。",
    "这是一张2D医学图像，但问题简单明了，我将直接提供诊断意见。",
    "由于是2D图像且分析需求直接，无需调用专家模型即可完成回答。",
]


def add_thoughts_actions(entry):
    entry = copy.deepcopy(entry)
    if "image" in entry:
        new_path = entry["image"]
        if not os.path.exists(new_path):
            print(f"{new_path} is not exist")
            return {}
        entry["image"] = new_path
    for idx, conv in enumerate(entry["conversations"]):
        if conv["from"] == "human":
            continue
        elif conv["from"] == "gpt":
            new_conv = {}
            new_conv["from"] = "gpt"
            new_conv["thoughts"] = random.choice(thoughts_options)
            new_conv["actions"] = []
            new_conv["value"] = conv["value"]
            entry["conversations"][idx] = new_conv

    return entry


def process_convs(src_json, img_path):
    dst_json = []
    for entry in tqdm(src_json):
        new_entry = add_thoughts_actions(entry, img_path)
        dst_json.append(new_entry)
        # break

    return dst_json


def main(args):
    with open(args.src_json_file, "r") as handler:
        src_json = json.load(handler)

    dst_json = process_convs(src_json)
    with open(args.dst_json_file, "w") as handler:
        json.dump(dst_json, handler, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_json_file", type=str, required=True)
    parser.add_argument("--dst_json_file", type=str, required=True)
    args = parser.parse_args()

    main(args)
    # if True:
    #     img_path = "/home/tx-deepocean/data1/jxq/data/processed_m3/PATH-VQA"
    #     src_json_file = "/home/tx-deepocean/data1/jxq/data/processed_m3/PATH-VQA/pathvqa_instruct_train.json"
    #     dst_json_file = "/home/tx-deepocean/data1/jxq/data/processed_m3/PATH-VQA/pathvqa_instruct_train_add_thoughts.json"

    #     with open(src_json_file, 'r') as handler:
    #         src_json = json.load(handler)

    #     dst_json = process_convs(src_json, img_path)
    #     with open(dst_json_file, "w") as handler:
    #         json.dump(dst_json, handler, indent=2, ensure_ascii=False)

    # if True:
    #     img_path = "/home/tx-deepocean/data1/jxq/data/processed_m3/RAD-VQA/imgs/"
    #     src_json_file = "/home/tx-deepocean/data1/jxq/data/processed_m3/RAD-VQA/radvqa_train_instruct.json"
    #     dst_json_file = "/home/tx-deepocean/data1/jxq/data/processed_m3/RAD-VQA/radvqa_train_instruct_add_thoughts.json"

    #     with open(src_json_file, 'r') as handler:
    #         src_json = json.load(handler)

    #     dst_json = process_convs(src_json, img_path)
    #     with open(dst_json_file, "w") as handler:
    #         json.dump(dst_json, handler, indent=2, ensure_ascii=False)

    # if True:
    #     img_path = "/home/tx-deepocean/data1/jxq/data/processed_m3/Slake1.0/imgs"
    #     src_json_file = "/home/tx-deepocean/data1/jxq/data/processed_m3/Slake1.0/slake_train_instruct.json"
    #     dst_json_file = "/home/tx-deepocean/data1/jxq/data/processed_m3/Slake1.0/slake_train_instruct_add_thoughts.json"

    #     with open(src_json_file, 'r') as handler:
    #         src_json = json.load(handler)

    #     dst_json = process_convs(src_json, img_path)
    #     with open(dst_json_file, "w") as handler:
    #         json.dump(dst_json, handler, indent=2, ensure_ascii=False)
