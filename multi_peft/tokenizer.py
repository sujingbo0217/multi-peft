# @Time : 10/31/23 3:21 PM
# @Author : Jingbo Su
# @File : tokenizer.py
# @Software : PyCharm

from transformers import AutoTokenizer
from .config import Tokens


class Tokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.vocab_size = self.tokenizer.vocab_size
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.unk_id = self.tokenizer.unk_token_id

        if self.pad_id is None and self.unk_id is not None:
            self.pad_id = self.unk_id

    def encode(self, data: str, bos: bool, eos: bool) -> Tokens:
        result = self.tokenizer.encode(data, add_special_tokens=False)
        if bos and self.bos_id is not None:
            result = [self.bos_id] + result
        if eos and self.eos_id is not None:
            result = result + [self.eos_id]
        return result

    def decode(self, data: Tokens) -> str:
        return self.tokenizer.decode(data)
