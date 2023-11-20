# @Time : 11/1/23 2:21 PM
# @Author : Jingbo Su
# @File : dispatcher.py
# @Software : PyCharm

import math
from typing import Dict, List, Any

from .agent import Agent
from .config import BatchInputConfig, MultiBatchInputConfig, Tokens
from .tokenizer import Tokenizer


class Dispatcher:
    def __init__(self, config: Dict[str, Any], tokenizer: Tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.lora_agents: List[Agent] = [Agent(
            tokenizer=self.tokenizer,
            agent_id=lora_conf.get('agent_id'),
            train_data_path=lora_conf.get('data'),
            val_set_size=lora_conf.get('val_set_size', -1),
            epoch_num=lora_conf.get('num_epochs'),
            test_data_path=lora_conf.get('test_data', None),
            train_batch_size=lora_conf.get('batch_size'),
            micro_batch_size=lora_conf.get('micro_batch_size'),
            test_batch_size=lora_conf.get('test_batch_size'),
            cutoff_len=config.get('cutoff_len'),
            group_by_length=config.get('group_by_length'),
        ) for lora_conf in config.get('lora')]
        # self.bottleneck_agents: List[Agent] = [Agent(tokenizer=self.tokenizer, **bottleneck_config) for bottleneck_config in config.get('bottleneck')]

    def get_train_data(self) -> MultiBatchInputConfig:
        lora_train_data = {}
        for agent in self.lora_agents:
            lora_train_data[agent.agent_id] = agent.get_batch_data()

        # bottleneck_train_data = {}
        # for agent in self.bottleneck_agents:
        #     bottleneck_train_data[agent.agent_id] = agent.get_batch_data()

        batch_seq_len: int = -1
        # Align batch token data
        for agent_id, train_data in lora_train_data.items():
            for data in train_data:
                batch_seq_len = max(batch_seq_len, len(data.tokens))

        prompts: List[str] = []
        batch_seq_len = math.ceil(batch_seq_len / 8) * 8
        expand_side: List[str] = []
        batch_tokens: List[Tokens] = []
        tokens_len_without_pad: List[int] = []
        lora_batch_input_config: List[BatchInputConfig] = []

        # Batch the all adapter data
        for agent_id in lora_train_data:
            train_data = lora_train_data.get(agent_id)
            for data in train_data:
                prompts.append(data.prompt)
                tokens: Tokens = data.tokens.copy()
                tokens_len_without_pad.append(len(tokens))
                # Get the pad token from lora config
                lora_config = None
                for config in self.config["lora"]:
                    if config.get("agent_id") == agent_id:
                        lora_config = config
                        break
                pad_side = lora_config.get("expand_side", "right")
                assert pad_side == "right" or pad_side == "left"
                # Pad the tokens to align
                while len(tokens) < batch_seq_len:
                    if pad_side == "right":
                        tokens.append(self.tokenizer.pad_id)
                    else:
                        tokens.insert(0, self.tokenizer.pad_id)
                expand_side.append(pad_side)
                batch_tokens.append(tokens)

            lora_batch_input_config.append(BatchInputConfig(agent_id=agent_id, batch_size=len(train_data)))

        return MultiBatchInputConfig(
            prompts=prompts,
            tokens=batch_tokens,
            batch_seq_len=batch_seq_len,
            batch_input_config_list=lora_batch_input_config,
            expand_side=expand_side,
            tokens_len_without_pad=tokens_len_without_pad,
        )

    def dispatch_completed_agent(self):
        self.lora_agents = [agent for agent in self.lora_agents if not agent.is_train_done()]
