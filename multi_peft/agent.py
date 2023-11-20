# @Time : 10/31/23 4:20 PM
# @Author : Jingbo Su
# @File : dispatcher.py
# @Software : PyCharm

import random
from dataclasses import dataclass
from typing import Dict, List

import datasets
from tqdm import tqdm

from .config import Tokens
from .tokenizer import Tokenizer


@dataclass
class DataType:
    prompt: str = ""
    tokens: Tokens = None


class Agent:
    agent_id: str = ""
    adapter_type: str = ""
    train_token_data: List[DataType] = None
    test_token_data: List[DataType] = None
    start_idx: int = 0
    epoch_cnt: int = 1

    def __init__(
            self,
            tokenizer: Tokenizer,
            agent_id: str,
            train_data_path: str,
            val_set_size: int,
            test_data_path: str,
            epoch_num: int,
            train_batch_size: int,
            micro_batch_size: int,
            test_batch_size: int,
            cutoff_len: int = 256,
            group_by_length: bool = True,
            expand_side: str = 'right',
            expand_token_id: int = 0
    ):
        self.tokenizer = tokenizer
        self.agent_id = agent_id
        self.train_data_path = train_data_path
        self.val_set_size = val_set_size
        self.test_data_path = test_data_path
        self.epoch_num = epoch_num
        self.train_batch_size = train_batch_size
        self.micro_batch_size = micro_batch_size
        self.test_batch_size = test_batch_size
        self.cutoff_len = cutoff_len
        self.group_by_length = group_by_length
        self.expand_side = expand_side
        self.expand_token_id = expand_token_id
        self.load_data()

    def __data_encoding(self, data: datasets.Dataset, is_train: bool = True) -> List[DataType]:
        return_data: List[DataType] = []
        for i, text in tqdm(enumerate(data), total=len(data), desc=f"Encoding text data {self.agent_id}"):
            prompt = self.__generate_prompt(text)
            if is_train:
                tokens = self.tokenizer.encode(prompt, bos=True, eos=True)
                if len(tokens) > self.cutoff_len:
                    tokens = tokens[:self.cutoff_len]
            else:
                tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
            return_data.append(DataType(prompt, tokens))
        if is_train and self.group_by_length:
            return_data.sort(key=lambda x: len(x.tokens), reverse=True)
        else:
            random.shuffle(return_data)
        return return_data

    def load_data(self) -> None:
        data = self.__load_dataset(self.train_data_path)
        if self.test_data_path is None:
            if self.val_set_size is None or self.val_set_size <= 0:
                self.train_token_data = self.__data_encoding(data=data.get('train'), is_train=True)
                self.test_token_data = []
            else:
                train_and_val_data = data.get('train').train_test_split(test_size=self.val_set_size)
                self.train_token_data = self.__data_encoding(data=train_and_val_data.get('train'), is_train=True)
                self.test_token_data = self.__data_encoding(data=train_and_val_data.get('test'), is_train=False)
        else:
            pass

    def get_batch_data(self) -> List[DataType]:
        if self.start_idx + self.micro_batch_size >= len(self.train_token_data) and self.start_idx == 0:
            raise 'Batch size is greater than training data length.'
        return_data = self.train_token_data[self.start_idx: self.start_idx + self.micro_batch_size]
        self.start_idx += self.micro_batch_size
        if self.start_idx >= len(self.train_token_data):
            self.start_idx = 0
            self.epoch_cnt += 1
        return return_data

    def is_train_done(self):
        return False if self.epoch_cnt <= self.epoch_num else True

    @staticmethod
    def __load_dataset(data_path: str):
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            return datasets.load_dataset("json", data_files=data_path)
        else:
            return datasets.load_dataset(data_path)

    @staticmethod
    def __generate_prompt(data: Dict):
        if data.get('input') is not None:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                    ### Instruction:
                    {data.get('instruction')}

                    ### Input:
                    {data.get('input')}

                    ### Response:
                    {data.get('output')}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                    ### Instruction:
                    {data.get('instruction')}

                    ### Response:
                    {data.get('output')}"""
