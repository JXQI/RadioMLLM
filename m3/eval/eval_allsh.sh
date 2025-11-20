#!/bin/bash
# export PROJECT_PATH="/home/tx-deepocean/data1/jxq/code/VLM-Radiology-Agent-Framework"
# export MODEL_PATH="/home/tx-deepocean/data1/jxq/code/VLM-Radiology-Agent-Framework/m3/demo/utils/Llama3-VILA-M3-8B"
# export OUTPUT_FOLDER_NAME="/home/tx-deepocean/data1/jxq/code/VLM-Radiology-Agent-Framework/m3/eval/eval_results"
# export CONV_MODE="llama_3"

export PROJECT_PATH="/home/tx-deepocean/data1/jxq/code/LLaVA-Med"
export MODEL_PATH="/home/tx-deepocean/data1/jxq/code/LLaVA-Med/outputs/models/llava-med-v1.5-mistral-7b-lora-mistral_instruct-heart2-chest2_slake1_path1_rad1_v5"
export OUTPUT_FOLDER_NAME="/home/tx-deepocean/data1/jxq/code/LLaVA-Med/eval/heart2-chest2_slake1_path1_rad1_v5"
export CONV_MODE="mistral_instruct"

export CUDA_VISIBLE_DEVICES=0,1,2,3
sh eval_pathvqa.sh $MODEL_PATH $OUTPUT_FOLDER_NAME $CONV_MODE
sh eval_radvqa.sh $MODEL_PATH $OUTPUT_FOLDER_NAME $CONV_MODE
sh eval_slakevqa.sh $MODEL_PATH $OUTPUT_FOLDER_NAME $CONV_MODE
# sh eval_expert.sh $MODEL_PATH $OUTPUT_FOLDER_NAME $CONV_MODE

