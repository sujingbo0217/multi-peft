# @Time : 11/1/23 11:19 PM
# @Author : Jingbo Su
# @File : modeling_llama.py
# @Software : PyCharm

from typing import List, Dict, Tuple, Any, Optional

import einops
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import xformers.ops
import xformers.ops.fmha.attn_bias
from transformers import AutoModelForCausalLM, OPTForCausalLM, LlamaForCausalLM

from .checkpoint import CheckpointRecomputeFunction
from .config import MultiBatchInputConfig, LlamaConfig
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


def precompute_rope_angle(dim: int, seq_len: int, device: str, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    angles = 1.0 / (theta ** (torch.arange(0, dim, 2).to(device)[: (dim // 2)].to(torch.float) / dim))
    seq = torch.arange(seq_len, device=angles.device)
    emb = torch.outer(seq, angles).float()
    emb = einops.repeat(emb, "... n -> ... (n r)", r=2)
    # cos(angle), sin(angle)
    return emb.cos().to(torch.float32), emb.sin().to(torch.float32)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = einops.rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return einops.rearrange(x, "... d r -> ... (d r)")


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim).reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, angle: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    # data shape is: batch_size * max_seq_len * n_head * n_dim
    _, max_seq_len, _, dim_head = xq.shape

    cos = angle[0][:max_seq_len].view(max_seq_len, 1, dim_head)
    sin = angle[1][:max_seq_len].view(max_seq_len, 1, dim_head)

    xq = (xq * cos) + (rotate_half(xq) * sin)
    xk = (xk * cos) + (rotate_half(xk) * sin)
    return xq, xk


@torch.jit.script
def apply_rotary_emb_to_one(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: batch_size, seq_len, num_head, head_dim
    _, seq_len, num_head, _ = x.size(0), x.size(1), x.size(2), x.size(3)

    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]

    # truncate to support variable sizes
    rope_cache = rope_cache[:seq_len]

    xshaped = x.reshape(-1, seq_len, num_head, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, seq_len, 1, xshaped.size(3), 2)

    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] -
            xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] +
            xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl

    @staticmethod
    def forward_impl(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()

        return cache

    def forward(self, max_seq_len, ):
        return self.forward_impl(max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device)


class RMSNorm:
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        self.norm_eps = eps
        self.weight = weight

    def _norm(self, data: torch.Tensor) -> torch.Tensor:
        return data * torch.rsqrt(torch.tensor(+ self.norm_eps))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        input_dtype = data.dtype
        v = data.to(torch.float32).pow(2).mean(-1, keepdim=True)
        data = data * torch.rsqrt(v + self.norm_eps)

        return (self.weight * data).to(input_dtype)


class Transformer:
    def __init__(self, layer_id: int, config: LlamaConfig):
        # attention
        self.wq: Optional[LoraLinear] = None  # dim * dim
        self.wk: Optional[LoraLinear] = None  # dim * dim
        self.wv: Optional[LoraLinear] = None  # dim * dim
        self.wo: Optional[LoraLinear] = None  # dim * dim
        # feed forward
        self.w1: Optional[LoraLinear] = None  # also gate FNN * dim
        self.w2: Optional[LoraLinear] = None  # also down dim * FNN
        self.w3: Optional[LoraLinear] = None  # also up   FNN * dim
        # norm
        self.attention_norm: Optional[RMSNorm] = None  # dim
        self.ffn_norm: Optional[RMSNorm] = None  # dim
        # other arg
        self.layer_id = layer_id
        self.norm_eps = config.norm_eps
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.hidden_size // config.n_heads

    def init_lora_layer_weight(
            self,
            adapter_name: str,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            target: Dict[str, bool],
            weight: Optional[Dict[str, torch.Tensor]]
    ):
        linear_layer_list = [self.wk, self.wq, self.wv, self.wo, self.w1, self.w2, self.w3]
        linear_layer_name_list = ["k_proj", "q_proj", "v_proj", "o_proj", "w1_proj", "w2_proj", "w3_proj"]

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

    # @torch.compile
    def forward(
            self,
            data: torch.Tensor,
            mask: torch.Tensor,
            rope_angle: Tuple[torch.Tensor, torch.Tensor],
            input_args: MultiBatchInputConfig
    ):
        batch_size, max_seq_len, _ = data.shape
        attention_norm_data = self.attention_norm.forward(data)

        xq = self.wq.forward(attention_norm_data, input_args)
        xk = self.wk.forward(attention_norm_data, input_args)
        xv = self.wv.forward(attention_norm_data, input_args)

        # convert shape to multi head
        xq = xq.view(batch_size, max_seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, max_seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, max_seq_len, self.n_kv_heads, self.head_dim)

        # apply rotary embedding
        xq, xk = apply_rotary_emb(xq, xk, rope_angle)

        # for llama2 need to repeat the heads
        # before dim: batch_size, seq_len, n_kv_head, head_dim
        # after dim: batch_size, seq_len, n_head, head_dim
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        attention_score = xformers.ops.memory_efficient_attention(xq, xk, xv, mask)
        attention_score = attention_score.view(batch_size, max_seq_len, -1)

        # get output attention score
        data = data + self.wo.forward(attention_score, input_args)

        # feed forward fully connected
        score_norm_data = self.ffn_norm.forward(data)
        w1 = self.w1.forward(score_norm_data, input_args)
        w3 = self.w3.forward(score_norm_data, input_args)

        data = data + self.w2.forward(F.silu(w1) * w3, input_args)
        return data


class LlamaModel(BaseModelMixin):
    model_type = 'llama'

    def __init__(self, config: LlamaConfig):
        # weight
        self.token_embedding: Optional[torch.Tensor] = None

        self.layers: List[Transformer] = []
        for layer_id in range(config.n_layers):
            self.layers.append(Transformer(layer_id, config))

        self.norm: Optional[RMSNorm] = None  # dim
        self.output: Optional[torch.Tensor] = None  # vocab size * dim

        # cos and sin
        self.rope_angle: Tuple[torch.Tensor, torch.Tensor] = precompute_rope_angle(
            config.hidden_size // config.n_heads, config.max_seq_len, config.device)

        self.norm_eps = config.norm_eps
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
        data = F.embedding(tokens, self.token_embedding, padding_idx=self.pad_token_id).requires_grad_(True)

        def create_forward_for_checkpoint(module: Transformer):
            def forward_for_checkpoint(*inputs_):
                return module.forward(*inputs_)

            return forward_for_checkpoint

        for layer in self.layers:
            # use CheckpointOffloadFunction to use offload mode
            if inputs.is_inference:
                data = layer.forward(data, mask, self.rope_angle, inputs)
            else:
                data = CheckpointRecomputeFunction.apply(create_forward_for_checkpoint(layer), data, mask, self.rope_angle, inputs)

        data = self.norm.forward(data)
        data @= self.output.transpose(0, 1)
        return data

    def init_lora_weight(
            self,
            adapter_name: str,
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
    ) -> 'LlamaModel':
        if bits in [4, 8]:
            if log_fn is not None:
                log_fn('Loading model with quantization, bits = %i' % bits)
            from transformers import BitsAndBytesConfig
            compute_dtype = (torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float32))
            if model_type == 'llama':
                llama_model = LlamaForCausalLM.from_pretrained(
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
            elif model_type == 'opt':
                llama_model = OPTForCausalLM.from_pretrained(
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
                llama_model = AutoModelForCausalLM.from_pretrained(
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
            if model_type == 'llama':
                llama_model = LlamaForCausalLM.from_pretrained(path, device_map=device, torch_dtype=torch.float32)
            elif model_type == 'opt':
                llama_model = OPTForCausalLM.from_pretrained(path, device_map=device, torch_dtype=torch.float32)
            else:
                llama_model = AutoModelForCausalLM.from_pretrained(path, device_map=device, torch_dtype=torch.float32)

        for param in llama_model.parameters():
            param.requires_grad = False

        llama_args = LlamaConfig(
            hidden_size=llama_model.config.hidden_size,
            n_heads=llama_model.config.num_attention_heads,
            n_layers=llama_model.config.num_hidden_layers,
            vocab_size=llama_model.config.vocab_size,
            max_seq_len=4096 if not hasattr(llama_model.config, "max_sequence_length") else llama_model.config.max_sequence_length,
            pad_token_id=-1,
            device=device,
        )
        llama_args.n_kv_heads = llama_args.n_heads \
            if not hasattr(llama_model.config, "num_key_value_heads") else llama_model.config.num_key_value_heads
        llama_args.norm_eps = llama_model.config.rms_norm_eps \
            if hasattr(llama_model.config, "rms_norm_eps") else None

        model = LlamaModel(llama_args)

        model.token_embedding = llama_model.model.embed_tokens.weight.to(device=device).requires_grad_(False)
        model.output = llama_model.lm_head.weight.to(dtype=torch.float32, device=device).requires_grad_(False)
        model.norm = RMSNorm(llama_model.model.norm.weight.to(device=device).requires_grad_(False), model.norm_eps)

        for idx, layer in enumerate(llama_model.model.layers):
            model.layers[idx].wq = LoraLinear(layer.self_attn.q_proj.requires_grad_(False), device=device)
            model.layers[idx].wk = LoraLinear(layer.self_attn.k_proj.requires_grad_(False), device=device)
            model.layers[idx].wv = LoraLinear(layer.self_attn.v_proj.requires_grad_(False), device=device)
            model.layers[idx].wo = LoraLinear(layer.self_attn.o_proj.requires_grad_(False), device=device)
            model.layers[idx].w1 = LoraLinear(layer.mlp.gate_proj.requires_grad_(False), device=device)
            model.layers[idx].w2 = LoraLinear(layer.mlp.down_proj.requires_grad_(False), device=device)
            model.layers[idx].w3 = LoraLinear(layer.mlp.up_proj.requires_grad_(False), device=device)
            model.layers[idx].attention_norm = RMSNorm(layer.input_layernorm.weight.to(device=device).requires_grad_(False), model.norm_eps)
            model.layers[idx].ffn_norm = RMSNorm(layer.post_attention_layernorm.weight.to(device=device).requires_grad_(False), model.norm_eps)
        return model

    def get_train_params(self, config: Dict[str, Any]) -> Dict[str, List[torch.Tensor]]:
        train_params = {}
        for transformer_layer in self.layers:
            for lora_config in config["lora"]:
                adapter_name = lora_config["agent_id"]
                if adapter_name not in train_params:
                    train_params[adapter_name] = []

                lora_layer_list = [transformer_layer.wq.multi_lora_model_dict, transformer_layer.wk.multi_lora_model_dict,
                                   transformer_layer.wv.multi_lora_model_dict, transformer_layer.wo.multi_lora_model_dict,
                                   transformer_layer.w1.multi_lora_model_dict, transformer_layer.w2.multi_lora_model_dict,
                                   transformer_layer.w3.multi_lora_model_dict]

                for lora_layer in lora_layer_list:
                    if adapter_name in lora_layer:
                        train_params[adapter_name].append(lora_layer[adapter_name].lora_a)
                        train_params[adapter_name].append(lora_layer[adapter_name].lora_b)

        return train_params
