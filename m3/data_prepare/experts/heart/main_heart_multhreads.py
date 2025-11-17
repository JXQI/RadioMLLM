# import os
# from langchain.llms import OpenAI
# os.environ["OPENAI_API_KEY"] = "sk-61f79ca08fa44bd3be0f24d82614a02d"

import argparse
import concurrent.futures
import os
import threading

# llm = OpenAI(model_name="qwen-plus", n=2, best_of=2)
# result = llm("给我讲个笑话")
# print(result)
from cmath import inf

import yaml
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

os.environ["DEEPSEEK_API_KEY"] = "sk-b4c46f3065484299b84398c5d050263f"

import copy
import json
import logging
import random
import sys
from typing import Any, Dict, List

from langchain.chains.base import Chain
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from data_utils import write_json
from utils.format import filter_valid_conversations


class GenRetrievalQA:
    def __init__(self, config_file: str, knowleage_file: str, has_memory: bool = False):
        self.llm = self.init_llm()
        self.retriever = self.init_retriever(knowleage_file)
        self.system_message = self.init_system_message(config_file)
        if has_memory:
            self.app = self.init_chatbot(self.llm, self.system_message)
        else:
            self.app = self.init_chain(self.system_message)
        self.out_parser = JsonOutputParser()

    def init_llm(self):
        llm = init_chat_model("deepseek-chat", model_provider="deepseek")

        return llm

    def init_retriever(self, knowleage_file):
        if (knowleage_file is None) or (not os.path.exists(knowleage_file)):
            return None
        loader = JSONLoader(knowleage_file, jq_schema=".[]")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        logging.info(f"文档总块数: {len(docs)}")
        embedding_model = DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key="sk-61f79ca08fa44bd3be0f24d82614a02d",
        )
        db = FAISS.from_documents(docs, embedding_model)
        retriever = db.as_retriever(search_kwargs={"k": 3})

        return retriever

    def init_system_message(self, config_file):
        with open(config_file, "r", encoding="utf-8") as file:
            config = file.read()

        return config

    def init_chatbot(self, model, system_message):
        workflow = StateGraph(state_schema=MessagesState)

        def call_model(state: MessagesState):
            response = model.invoke(state["messages"])
            return {"messages": response}

        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)

        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        self.config = {"configurable": {"thread_id": "abc234"}}
        output = app.invoke({"messages": SystemMessage(system_message)}, self.config)
        output["messages"][-1].pretty_print()
        logging.info(output)

        return app

    def init_chain(self, system_message):
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_message),
                HumanMessagePromptTemplate.from_template("{user_input}"),
            ]
        )
        print("提示模板的必填输入变量：", prompt.input_variables)
        return prompt | self.llm

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs["query"]
        full_input = f"病灶检测结果: {query}"

        output = self.app.invoke({"user_input": full_input})

        return self.out_parser.parse(output.content)


class HeartDataSet:
    def __init__(self, bot, data_info_path: str, max_workers: int = 4):
        self.bot = bot
        self.max_workers = max_workers

        with open(data_info_path, "r") as handle:
            self.data_info = json.load(handle)
        self.lock = threading.Lock()
        self.add_conversion = []

    def invoke(self, query):
        result = self.bot.invoke({"query": query})
        return result

    @classmethod
    def get_query(cls, lesion_report: List[Dict]):
        report = ""
        for item in lesion_report.values():
            report += item
            report += "\n"
        return report

    def _gen_conv(self, data_info, convs, idx):
        conversations = []
        for conv in convs:
            _slices_images = list(data_info["slices"].values())
            _entry = {
                "image": random.choice(_slices_images),
            }

            _entry.update(conv)

            conversations.append(copy.copy(_entry))
        print(f"线程 {threading.current_thread().name}：成功 idx={idx}")
        return conversations

    def _process_single_task(self, idx, info):
        """处理单个 data_info 任务，返回生成的 conversations"""
        try:
            output = self.invoke(info["lesion_report"])
            return self._gen_conv(info, output, idx)
        except Exception as e:
            print(
                f"线程 {threading.current_thread().name}：错误 idx={idx}, dicom={info['dicom']}, 错误={e}"
            )
            return []  # 出错时返回空列表，不影响整体结果

    def __call__(self, tasks):
        self.add_conversion = []
        tasks = [(idx, info) for idx, info in enumerate(tasks)]  # 剩下五十个当作测试集
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="HeartThread"
        ) as executor:
            # 提交所有任务，返回结果迭代器
            future_results = executor.map(
                lambda x: self._process_single_task(x[0], x[1]), tasks
            )

            for result in future_results:
                if result:
                    with self.lock:  # 加锁写入共享结果列表
                        self.add_conversion.extend(result)

        # 过滤掉不符合的对话
        self.add_conversion = filter_valid_conversations(self.add_conversion)
        return self.add_conversion


def main(args):

    bot = GenRetrievalQA(args.config_file, args.knowloge_file)
    heart_dataset = HeartDataSet(
        bot, data_info_path=args.data_info, max_workers=args.max_workers
    )

    data_infos = heart_dataset.data_info[-50:]
    split_idx = int(len(data_infos) * args.test_frac)

    out_train_file = args.out_fileprefix + "_train.json"
    out_test_file = args.out_fileprefix + "_test.json"

    test_conversions = heart_dataset(data_infos[0:split_idx])
    write_json(test_conversions, out_test_file)

    train_conversions = heart_dataset(data_infos[split_idx:])
    write_json(train_conversions, out_train_file)

    print(f"train_convs: {len(train_conversions)}, test_convs: {len(test_conversions)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_info", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--knowloge_file", type=str, default=None)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--out_fileprefix", type=str, required=True)
    parser.add_argument("--test_frac", type=float, default=0.5)
    args = parser.parse_args()

    main(args)
