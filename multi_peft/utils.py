# @Time : 11/2/23 11:27 PM
# @Author : Jingbo Su
# @File : utils.py
# @Software : PyCharm

import json
import os
from typing import Dict, Any

import torch

from . import BaseModelMixin


def save_lora_model(model: BaseModelMixin, config: Dict[str, Any], dir_suffix=""):
    model_type: str = model.model_type

    for lora_config in config["lora"]:
        lora_name = lora_config["agent_id"]
        lora_output_dir = os.path.join('experiments', lora_config["output"])
        if dir_suffix != "":
            lora_output_dir += os.sep + lora_config["output"] + "_" + dir_suffix

        if not os.path.exists(lora_output_dir):
            os.makedirs(lora_output_dir)

        lora_weight_dict = {}
        target_modules = []
        for i, transformer_layer in enumerate(model.layers):
            layer_prefix_name = "base_model.model.model.layers." + str(i) + "." + "self_attn."
            lora_layer_list, lora_layer_name_list = [], []
            
            if model_type == 'llama':
                lora_layer_list = [transformer_layer.wq, transformer_layer.wk,
                                   transformer_layer.wv, transformer_layer.wo,
                                   transformer_layer.w1, transformer_layer.w2,
                                   transformer_layer.w3]
                lora_layer_name_list = ["q_proj", "k_proj", "v_proj", "o_proj", "w1_proj", "w2_proj", "w3_proj"]
            elif model_type == 'opt':
                lora_layer_list = [transformer_layer.wq, transformer_layer.wk, transformer_layer.wv,
                                   transformer_layer.wo, transformer_layer.w1, transformer_layer.w2]
                lora_layer_name_list = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
            else:
                pass

            for j, lora_layer in enumerate(lora_layer_list):
                if lora_name in lora_layer.multi_lora_model_dict:
                    if lora_layer_name_list[j] not in target_modules:
                        target_modules.append(lora_layer_name_list[j])
                    lora_weight_dict[layer_prefix_name +
                                     f"{lora_layer_name_list[j]}.lora_A.weight"] = lora_layer.multi_lora_model_dict[lora_name].lora_a
                    lora_weight_dict[layer_prefix_name +
                                     f"{lora_layer_name_list[j]}.lora_B.weight"] = lora_layer.multi_lora_model_dict[lora_name].lora_b

        torch.save(lora_weight_dict, lora_output_dir + os.sep + "adapter_model.bin")

        adapter_config = {
            "lora_alpha": lora_config["alpha"],
            "lora_dropout": lora_config["dropout"],
            "r": lora_config["r"],
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "bias": "none",
            "target_modules": target_modules,
        }

        with open(lora_output_dir + os.sep + "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=4)
