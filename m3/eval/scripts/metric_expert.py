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


import argparse
import json
import os
import re
from typing import Any, Dict, List

import jsonlines
from langchain.chat_models import init_chat_model
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser

INSTRUCT_PROMPT = """我们邀请您就两位AI助手针对上方用户问题所给出的回答表现提供反馈。用户提问基于专家模型的病灶检出结果Context字段供您参考。\
                请评估其回答的实用性、相关性、准确性及细节详实度。每位助手将获得1到10分的总体评分，分数越高代表整体表现越佳。\
                以及请详细阐述您的评估依据，需避免潜在偏见，并确保回答的呈现顺序不会影响您的判断。\
                重要：输出必须为纯 JSON 格式，不要使用 Markdown 代码块或添加任何额外文本。请按以下格式输出：
  {{"Assistant1_Score": score, "Assistant2_Score": score, "Reason": "解释原因，必须用中文回答"}} """


class LLM:
    def __init__(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "sk-b4c46f3065484299b84398c5d050263f"
        self.llm = init_chat_model("deepseek-chat", model_provider="deepseek")
        self.chain = self.init_chain(INSTRUCT_PROMPT)
        # 移除 JsonOutputParser，使用自定义解析

    def init_chain(self, system_message):
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_message),
                HumanMessagePromptTemplate.from_template("{user_input}"),
            ]
        )
        print("提示模板的必填输入变量：", prompt.input_variables)
        return prompt | self.llm | StrOutputParser()

    def invoke_single(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """单个调用（保持原有功能）"""
        context = inputs["context"]
        answer1 = inputs["answer1"]
        answer2 = inputs["answer2"]
        full_input = f"context: {context} \n\nanswer1: {answer1}\n\nanswer2: {answer2}"

        output = self.chain.invoke({"user_input": full_input})
        print(output)
        return self._parse_output(output)

    def invoke_batch(self, inputs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """并行批量调用"""
        # 准备批量输入
        batch_inputs = []
        for i, inputs in enumerate(inputs_list):
            input_id = inputs.get("id", i)
            context = inputs.get("context", "")
            answer1 = inputs.get("answer1", "")
            answer2 = inputs.get("answer2", "")
            full_input = (
                f"context: {context} \n\nanswer1: {answer1}\n\nanswer2: {answer2}"
            )

            batch_inputs.append(
                {"user_input": full_input, "id": input_id, "context": context}
            )

        # 并行调用
        outputs = self.chain.batch(batch_inputs)

        # 解析所有结果
        results = []
        for i, output in enumerate(outputs):
            try:
                result = self._parse_output(output)
                result["id"] = inputs_list[i]["id"]
                result["context"] = inputs_list[i]["context"]
                result["answer1"] = inputs_list[i]["answer1"]
                result["answer2"] = inputs_list[i]["answer2"]
                results.append(result)
            except Exception as e:
                print(f"解析输出时出错: {e}")
                results.append(
                    {
                        "error": str(e),
                        "raw_output": output,
                        "id": batch_inputs[i]["id"],
                        "context": batch_inputs[i]["context"],
                    }
                )

        return results

    def _parse_output(self, output: str) -> Dict[str, Any]:
        """解析输出为 JSON"""
        try:
            # 尝试直接解析 JSON
            return json.loads(output)
        except json.JSONDecodeError:
            # 如果失败，尝试提取 JSON 部分
            json_match = re.search(r"\{.*\}", output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                raise ValueError(f"无法从输出中提取 JSON: {output}")


llm = LLM()


def get_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=True)
    parser.add_argument("--answers", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def load_jsonl(file_path):
    """Load a jsonl file."""
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data


def extract_gpt_values(json_data):
    """Extract the GPT values from the JSON data."""
    gpt_values = {}
    for item in json_data:
        try:
            id = item["image"]
            metrics = {}
            metrics["context"] = item["conversations"][2]["value"].strip()
            metrics["question"] = (
                item["conversations"][0]["value"].replace("<image>", "").strip().lower()
            )
            metrics["api_name"] = (
                item["conversations"][1]["actions"][0]["API_name"].strip().lower()
            )
            metrics["value"] = item["conversations"][-1]["value"].strip().lower()
            gpt_values[id] = metrics
        except Exception as e:
            print(e)
            print(item, "=====")
    return gpt_values


def extract_text_values(jsonl_data):
    """Extract the text values from the JSONL data."""
    text_values = {}
    for item in jsonl_data:
        try:
            id = item["image"]
            metrics = {}
            metrics["api_name"] = item["api_name"].strip().lower()
            metrics["value"] = item["value"].strip().lower()
            text_values[id] = metrics
        except Exception as e:
            print(item, f"结果不符合格式要求:{e}")

    return text_values


def calculate_accuracy(gpt_values, text_values):
    """Calculate the accuracy of the model."""
    correct = 0
    total = 0

    for id, gpt_value in gpt_values.items():
        if id in text_values:
            total += 1
            gpt_aip_name = gpt_value["api_name"]
            pred_aip_name = text_values[id]["api_name"]
            if gpt_aip_name == pred_aip_name:
                correct += 1

    return correct / total if total > 0 else 0


def calculate_matching_accuracy(gpt_values, text_values):
    """Calculate the accuracy of the model."""
    correct = 0
    total = 0

    batch_query = []
    for id, gpt_value in gpt_values.items():
        if id in text_values:
            total += 1
            query = {}
            query["id"] = id
            query["context"] = gpt_value["context"]
            query["answer1"] = gpt_value["value"]
            query["answer2"] = text_values[id]["value"]

            batch_query.append(query)

    output = llm.invoke_batch(batch_query)

    # 求平均得分
    count = len(output)
    answer1_score, answer2_score = 0, 0
    for id, response in enumerate(output):
        answer1_score += response["Assistant1_Score"]
        answer2_score += response["Assistant2_Score"]

    average_scores = {
        "Assistant1_Score": float(answer1_score) / count,
        "Assistant2_Score": float(answer2_score) / count,
    }
    return average_scores, output


def main():
    """Main function."""
    args = get_args()
    json_data = load_json(args.input)
    jsonl_data = load_jsonl(args.answers)

    gpt_values = extract_gpt_values(json_data)
    text_values = extract_text_values(jsonl_data)

    tool_accuracy = calculate_accuracy(gpt_values, text_values)
    average_scores, output_eval = calculate_matching_accuracy(gpt_values, text_values)

    output_eval_file = args.output.split(".")[0] + "_eval.jsonl"
    print(f"EXPERT {args.input} {args.answers}")
    print(f"EXPERT Accuracy: {tool_accuracy :.4f} saved to {args.output}")
    print(f"EXPERT Eval: {output_eval} saved to {output_eval_file}")

    average_scores.update({"action_accuracy": tool_accuracy})
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(average_scores, f, ensure_ascii=False, indent=2)
    with open(output_eval_file, "w", encoding="utf-8") as f:
        json.dump(output_eval, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
