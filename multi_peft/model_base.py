# @Time : 11/1/23 6:45 PM
# @Author : Jingbo Su
# @File : model_base.py
# @Software : PyCharm

from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional

import torch

from .config import MultiBatchInputConfig


class BaseModelMixin(metaclass=ABCMeta):
    vocab_size = None
    layers = None
    model_type = None

    @abstractmethod
    def forward(self, inputs: MultiBatchInputConfig):
        pass

    @abstractmethod
    def get_train_params(self, config: Dict[str, str]) -> Dict[str, List[torch.Tensor]]:
        pass

    @abstractmethod
    def init_lora_weight(
            self,
            adapter_name: str,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            target: Dict[str, bool],
            weight: Optional[Dict[str, torch.Tensor]]
    ):
        pass

    # TODO: def init_bottleneck_weight
