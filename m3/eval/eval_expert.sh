#!/bin/bash

#SBATCH --job-name=slakevqa_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00
#SBATCH --partition=interactive,interactive_singlenode,grizzly,polar,polar2,polar3,polar4
#SBATCH --dependency=singleton

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# set common env vars
# source set_env.sh

# if [[ $# -ne 3 ]]; then
#     print_usage
#     exit 1
# fi

export MODEL_PATH=$1
export OUTPUT_FOLDER_NAME=$2
export CONV_MODE=$3


# 心脏
DATA_PATH="/home/tx-deepocean/data2/jxq/data/mmedagent/processed/heart/expert_heart_conv_v3_1_test.json"
IMAGE_DIR="/home/tx-deepocean/data2/jxq/data/mmedagent/processed/heart/"
OUTPUT_ROOT="$OUTPUT_FOLDER_NAME/expert/heart"
OUTPUT_PATH="$OUTPUT_ROOT/outputs.jsonl"
RESULT_PATH="$OUTPUT_ROOT/results.json"


# python -m llava.eval.model_vqa_expert \
#         --model-path $MODEL_PATH \
#         --question-file $DATA_PATH \
#         --image-folder $IMAGE_DIR \
#         --answers-file $OUTPUT_PATH \
#         --num-chunks 1 \
#         --chunk-idx 0 \
#         --conv-mode $CONV_MODE
#         # --single-pred-prompt \
#         

# torchrun --nproc_per_node=4 -m llava.eval.model_vqa_expert \
#         --model-path $MODEL_PATH \
#         --question-file $DATA_PATH \
#         --image-folder $IMAGE_DIR \
#         --answers-file $OUTPUT_PATH \
#         --conv-mode $CONV_MODE
#         # --single-pred-prompt \
         

python $PROJECT_PATH/m3/eval/scripts/metric_expert.py \
    --input $DATA_PATH \
    --answers $OUTPUT_PATH \
    --output $RESULT_PATH 


# 胸肺
DATA_PATH="/home/tx-deepocean/data2/jxq/data/mmedagent/processed/chest/expert_chest_conv_v3_1_test.json"
IMAGE_DIR="/home/tx-deepocean/data2/jxq/data/mmedagent/processed/chest"
OUTPUT_ROOT="$OUTPUT_FOLDER_NAME/expert/chest"
OUTPUT_PATH="$OUTPUT_ROOT/outputs.jsonl"
RESULT_PATH="$OUTPUT_ROOT/results.json"


# python -m llava.eval.model_vqa_expert \
#         --model-path $MODEL_PATH \
#         --question-file $DATA_PATH \
#         --image-folder $IMAGE_DIR \
#         --answers-file $OUTPUT_PATH \
#         --num-chunks 1 \
#         --chunk-idx 0 \
#         --conv-mode $CONV_MODE
#         # --single-pred-prompt \
#         

# torchrun --nproc_per_node=4 -m llava.eval.model_vqa_expert \
#         --model-path $MODEL_PATH \
#         --question-file $DATA_PATH \
#         --image-folder $IMAGE_DIR \
#         --answers-file $OUTPUT_PATH \
#         --conv-mode $CONV_MODE
#         # --single-pred-prompt \
         

# python $PROJECT_PATH/m3/eval/scripts/metric_expert.py \
#     --input $DATA_PATH \
#     --answers $OUTPUT_PATH \
#     --output $RESULT_PATH 

