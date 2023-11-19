# @Time : 10/31/23 3:58 PM
# @Author : Jingbo Su
# @File : lora.py
# @Software : PyCharm

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MultiBatchInputConfig


class LoraModel:
    def __init__(self, r: int, alpha: int, dropout: float):
        self.lora_a: Optional[torch.Tensor] = None
        self.lora_b: Optional[torch.Tensor] = None
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.scaling: float = alpha / r

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        _data = F.dropout(data, self.dropout)
        _data @= self.lora_a.transpose(0, 1)
        _data @= self.lora_b.transpose(0, 1)
        _data *= self.scaling
        return _data


class LoraLinear:
    def __init__(self, weight: nn.Module, device: str = None):
        self.device = device if device is not None else weight.device
        self.weight = weight
        self.weight.to(self.device)
        self.multi_lora_model_dict: Dict[str, LoraModel] = {}

    def init_weight(self, agent_id: str, r: int, alpha: int, dropout: float,
                    lora_a: Optional[torch.Tensor] = None, lora_b: Optional[torch.Tensor] = None) -> None:

        if agent_id not in self.multi_lora_model_dict:
            self.multi_lora_model_dict[agent_id] = LoraModel(r, alpha, dropout)

        out_features, in_features = self.weight.weight.size()

        if lora_a is not None:
            self.multi_lora_model_dict.get(agent_id).lora_a = lora_a.to(
                device=self.device).to(torch.float32).requires_grad_(True)
        else:
            self.multi_lora_model_dict.get(agent_id).lora_a = torch.zeros(
                size=(r, in_features), device=self.device, requires_grad=True, dtype=torch.float32)
            torch.nn.init.kaiming_normal_(self.multi_lora_model_dict.get(agent_id).lora_a, a=math.sqrt(5))

        if lora_b is not None:
            self.multi_lora_model_dict.get(agent_id).lora_b = lora_b.to(
                device=self.device).to(torch.float32).requires_grad_(True)
        else:
            self.multi_lora_model_dict.get(agent_id).lora_b = torch.zeros(
                size=(out_features, r), device=self.device, requires_grad=True, dtype=torch.float32)

    def forward(self, data: torch.Tensor, inputs: MultiBatchInputConfig) -> torch.Tensor:
        result = self.weight.forward(data)
        idx = 0
        for batch_input in inputs.batch_input_config_list:
            agent_id = batch_input.agent_id
            batch_size = batch_input.batch_size
            result[idx: idx + batch_size] += self.multi_lora_model_dict[agent_id].forward(data=data[idx: idx + batch_size])
            idx += batch_size
            return result
