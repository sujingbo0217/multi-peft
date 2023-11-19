# @Time : 10/31/23 3:18 PM
# @Author : Jingbo Su
# @File : config.py
# @Software : PyCharm

from dataclasses import dataclass
from typing import List

Tokens = List[int]


@dataclass
class LlamaConfig:
    vocab_size: int = None
    hidden_size: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 32
    norm_eps: float = 1e-06
    pad_token_id: int = None
    max_seq_len: int = 2048
    device: str = ""


@dataclass
class OPTConfig:
    vocab_size: int = None
    hidden_size: int = 768
    n_layers: int = 12
    n_heads: int = 12
    pad_token_id: int = None
    device: str = ""


@dataclass
class BatchInputConfig:
    agent_id: str = ""
    batch_size: int = 0


@dataclass
class MultiBatchInputConfig:
    prompts: List[str] = ""
    tokens: List[Tokens] = None
    batch_seq_len: int = None
    batch_input_config_list: List[BatchInputConfig] = None
    expand_side: List[str] = None
    tokens_len_without_pad: Tokens = None
    is_inference: bool = False
