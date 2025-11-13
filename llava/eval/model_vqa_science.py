import argparse
import torch
import os
import itertools
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle, parse_tool_output
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from llava.add_utils import io
from llava.add_utils import distributed as dist

from PIL import Image
import math
from transformers import set_seed, logging

logging.set_verbosity_error()



def eval_model(args):
    set_seed(0)
    if args.num_chunks is None:
        dist.init()
        torch.cuda.set_device(dist.local_rank())
        world_size, global_rank = dist.size(), dist.rank()
    else:
        world_size, global_rank = args.num_chunks, args.chunk_idx
    print(world_size, global_rank)
    instances = io.load(args.question_file)[global_rank::world_size]

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name, model_path, args.model_base, args.question_file, "+++")
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    outputs = []
    for instance in tqdm(instances, disable=global_rank != 0):
        try:
            question = instance["conversations"][0]["value"]
            question = question.replace("<image>", "").strip()
            if args.single_pred_prompt:
                question = question + "\n" + "Answer with the option's letter from the given choices directly."
            
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + question

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            image = [Image.open(os.path.join(args.image_folder, instance["image"]))] if "image" in instance else []
            image_tensor = process_images(image, image_processor, model.config)[0]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=context_len,
                    use_cache=True)

            response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(instance, "---", response)
            matches = parse_tool_output(response)
            outputs.append(
                {"question_id": instance["id"], "prompt": instance["conversations"][0]["value"], "text": matches[0][2].strip()}
            )
        except Exception as e:
            print(e)
            print(instance)

    # Gather outputs and save
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = list(itertools.chain(*outputs))

    io.save(args.answers_file, outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/tx-deepocean/data1/jxq/code/LLaVA-Med/llava-med-v1.5-mistral-7b-lora-mistral_instruct-heart2-chest2_slake1_path1_rad1_v4")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/home/tx-deepocean/data1/jxq/data/processed_m3/RAD-VQA/imgs")
    parser.add_argument("--question-file", type=str, default="/home/tx-deepocean/data1/jxq/data/processed_m3/RAD-VQA/radvqa_test_instruct.json")
    parser.add_argument("--answers-file", type=str, default="/home/tx-deepocean/data1/jxq/code/LLaVA-Med/eval/heart2-chest2_slake1_path1_rad1_v4/radvqa_test_instruct_answer.json")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--conv-mode", type=str, default="mistral_instruct")
    parser.add_argument("--num-chunks", type=int)
    parser.add_argument("--chunk-idx", type=int)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
