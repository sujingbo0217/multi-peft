# @Time : 11/18/23 1:16 PM
# @Author : Jingbo Su
# @File : modeling_opt.py
# @Software : PyCharm


from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import xformers.ops
import xformers.ops.fmha.attn_bias
from transformers import OPTForCausalLM

from .checkpoint import CheckpointRecomputeFunction
from .config import MultiBatchInputConfig, OPTConfig
from .lora import LoraLinear
from .model_base import BaseModelMixin


def precompute_mask(inputs: MultiBatchInputConfig, n_head: int, device: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    mask = torch.full((len(inputs.prompts), n_head, inputs.batch_seq_len, inputs.batch_seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1).to(torch.float32).cuda(device)

    for idx, _ in enumerate(inputs.prompts):
        zero_len = inputs.tokens_len_without_pad[idx]
        inf_len = inputs.batch_seq_len - zero_len
        expand_side = inputs.expand_side[idx]
        if expand_side == "right":
            mask[idx] += torch.tensor([0] * zero_len + [float("-inf")] * inf_len).expand(inputs.batch_seq_len, inputs.batch_seq_len).cuda(device)
        else:
            mask[idx] += torch.tensor([float("-inf")] * inf_len + [0] * zero_len).expand(inputs.batch_seq_len, inputs.batch_seq_len).cuda(device)
    return mask.to(dtype)


class Transformer:
    def __init__(self, layer_id: int, config: OPTConfig):
        # attention layer
        self.wq: Optional[LoraLinear] = None  # dim * dim
        self.wk: Optional[LoraLinear] = None  # dim * dim
        self.wv: Optional[LoraLinear] = None  # dim * dim
        self.wo: Optional[LoraLinear] = None  # dim * dim

        # fully connected layer
        self.w1: Optional[LoraLinear] = None  # dim * ffn_dim
        self.w2: Optional[LoraLinear] = None  # ffn_dim * dim

        # layer norm
        self.self_attn_layer_norm: Optional[nn.LayerNorm] = None  # dim
        self.final_layer_norm: Optional[nn.LayerNorm] = None  # dim

        # other arg
        self.layer_id = layer_id
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_size // config.n_heads
        self.scaling = self.head_dim ** (-0.5)

    def init_lora_layer_weight(
            self,
            adapter_name: str,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            target: Dict[str, bool],
            weight: Optional[Dict[str, torch.Tensor]]
    ):
        linear_layer_list = [self.wk, self.wq, self.wv, self.wo, self.w1, self.w2]
        linear_layer_name_list = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]

        for idx, layer_name in enumerate(linear_layer_name_list):
            if layer_name in target and target[layer_name]:
                lora_a = None
                lora_b = None
                if weight is not None:
                    lora_a_name = f"base_model.model.model.layers.{self.layer_id}.self_attn.{layer_name}.lora_A.weight"
                    lora_b_name = f"base_model.model.model.layers.{self.layer_id}.self_attn.{layer_name}.lora_B.weight"
                    if lora_a_name not in weight:
                        raise f"can not found the layer {lora_a_name} in model"
                    if lora_b_name not in weight:
                        raise f"can not found the layer {lora_b_name} in model"
                    lora_a = weight[lora_a_name]
                    lora_b = weight[lora_b_name]

                linear_layer_list[idx].init_weight(adapter_name, r, lora_alpha, lora_dropout, lora_a, lora_b)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2).contiguous()

    # @torch.compile
    def forward(
            self,
            data: torch.Tensor,
            mask: torch.Tensor,
            input_args: MultiBatchInputConfig
    ):
        batch_size, max_seq_len, _ = data.size()
        attn_norm_data = self.self_attn_layer_norm(data)

        xq = self.wq.forward(attn_norm_data, input_args) * self.scaling
        xk = self._shape(self.wk.forward(attn_norm_data, input_args), -1, batch_size)
        xv = self._shape(self.wv.forward(attn_norm_data, input_args), -1, batch_size)

        proj_shape = (batch_size * self.n_heads, -1, self.head_dim)
        xq = self._shape(xq, max_seq_len, batch_size).view(*proj_shape)
        xk = xk.view(*proj_shape)
        xv = xv.view(*proj_shape)

        attn_weights = xformers.ops.memory_efficient_attention(xq, xk, xv, mask)
        attn_weights = attn_weights.view(batch_size, max_seq_len, -1)

        # get output attention score
        data = data + self.wo.forward(attn_weights, input_args)

        # fully connected
        norm_data = self.final_layer_norm(data)
        w1 = self.w1.forward(norm_data, input_args)
        data = data + self.w2.forward(F.relu(w1), input_args)

        return data


class OPTModel(BaseModelMixin):
    model_type = 'opt'

    def __init__(self, config: OPTConfig):
        # weight
        self.embed_tokens: Optional[torch.Tensor] = None
        self.layers: List[Transformer] = []
        for layer_id in range(config.n_layers):
            self.layers.append(Transformer(layer_id, config))
        self.output: Optional[torch.Tensor] = None  # dim * vocab_size

        self.device = config.device
        self.n_heads = config.n_heads
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.hidden_size = config.hidden_size

        # need to set
        self.eos_token_id = -1

    # train model or inference model: output is probs
    def forward(self, inputs: MultiBatchInputConfig) -> torch.Tensor:
        tokens = torch.tensor(inputs.tokens, dtype=torch.int64).to(self.device)

        # only for train
        mask = precompute_mask(inputs, self.n_heads, self.device)
        data = F.embedding(tokens, self.embed_tokens, padding_idx=self.pad_token_id).requires_grad_(True)

        def create_forward_for_checkpoint(module: Transformer):
            def forward_for_checkpoint(*inputs_):
                return module.forward(*inputs_)

            return forward_for_checkpoint

        for layer in self.layers:
            # use CheckpointOffloadFunction to use offload mode
            if inputs.is_inference:
                data = layer.forward(data, mask, inputs)
            else:
                data = CheckpointRecomputeFunction.apply(create_forward_for_checkpoint(layer), data, mask, inputs)
        data @= self.output.transpose(0, 1)

        return data

    def init_lora_weight(
            self, adapter_name: str,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            target: Dict[str, bool],
            weight: Optional[Dict[str, torch.Tensor]]
    ):
        for transformer_layer in self.layers:
            transformer_layer.init_lora_layer_weight(adapter_name, r, lora_alpha, lora_dropout, target, weight)

    @classmethod
    def from_pretrained(
            cls,
            path: str,
            model_type: str,
            device: str,
            bits: int = None,
            fp16: bool = True,
            bf16: bool = True,
            double_quant: bool = True,
            quant_type: str = 'nf4',
            log_fn=None,
    ) -> 'OPTModel':
        if bits in [4, 8]:
            if log_fn is not None:
                log_fn('Loading model with quantization, bits = %i' % bits)
            from transformers import BitsAndBytesConfig
            compute_dtype = (torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float32))
            if model_type == 'opt':
                opt_model = OPTForCausalLM.from_pretrained(
                    path,
                    load_in_4bit=(bits == 4),
                    load_in_8bit=(bits == 8),
                    device_map=device,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=(bits == 4),
                        load_in_8bit=(bits == 8),
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_use_double_quant=double_quant,
                        bnb_4bit_quant_type=quant_type,
                    ),
                    torch_dtype=(torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32))
                )
            else:
                raise ValueError('model_type must be opt.')
        else:
            if model_type == 'opt':
                opt_model = OPTForCausalLM.from_pretrained(path, device_map=device, torch_dtype=torch.float32)
            else:
                raise ValueError('model_type must be opt.')

        for param in opt_model.parameters():
            param.requires_grad = False

        opt_args = OPTConfig(
            hidden_size=opt_model.config.hidden_size,
            n_heads=opt_model.config.num_attention_heads,
            n_layers=opt_model.config.num_hidden_layers,
            vocab_size=opt_model.config.vocab_size,
            pad_token_id=-1,
            device=device,
        )

        model = OPTModel(opt_args)
        model.embed_tokens = opt_model.model.decoder.embed_tokens.weight.to(device=device).requires_grad_(False)
        model.output = opt_model.lm_head.weight.to(dtype=torch.float32, device=device).requires_grad_(False)

        for idx, layer in enumerate(opt_model.model.decoder.layers):
            model.layers[idx].wq = LoraLinear(layer.self_attn.q_proj.requires_grad_(False), device=device)
            model.layers[idx].wk = LoraLinear(layer.self_attn.k_proj.requires_grad_(False), device=device)
            model.layers[idx].wv = LoraLinear(layer.self_attn.v_proj.requires_grad_(False), device=device)
            model.layers[idx].wo = LoraLinear(layer.self_attn.out_proj.requires_grad_(False), device=device)
            model.layers[idx].w1 = LoraLinear(layer.fc1.requires_grad_(False), device=device)
            model.layers[idx].w2 = LoraLinear(layer.fc2.requires_grad_(False), device=device)
            model.layers[idx].self_attn_layer_norm = layer.self_attn_layer_norm.weight.to(device=device).requires_grad_(False)
            model.layers[idx].final_layer_norm = layer.final_layer_norm.weight.to(device=device).requires_grad_(False)

        return model

    def get_train_params(self, config: Dict[str, Any]) -> Dict[str, List[torch.Tensor]]:
        train_params = {}
        for transformer_layer in self.layers:
            for lora_config in config["lora"]:
                adapter_name = lora_config["agent_id"]
                if adapter_name not in train_params:
                    train_params[adapter_name] = []

                lora_layer_list = [
                    transformer_layer.wq.multi_lora_model_dict, transformer_layer.wk.multi_lora_model_dict,
                    transformer_layer.wv.multi_lora_model_dict, transformer_layer.wo.multi_lora_model_dict,
                    transformer_layer.w1.multi_lora_model_dict, transformer_layer.w2.multi_lora_model_dict,
                ]

                for lora_layer in lora_layer_list:
                    if adapter_name in lora_layer:
                        train_params[adapter_name].append(lora_layer[adapter_name].lora_a)
                        train_params[adapter_name].append(lora_layer[adapter_name].lora_b)

        return train_params
