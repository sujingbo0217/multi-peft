# @Time : 11/2/23 1:00 AM
# @Author : Jingbo Su
# @File : train.py
# @Software : PyCharm

import argparse
import datetime
import json
import random
import time
from threading import Thread
from typing import Dict, Tuple, List

import GPUtil
import torch

import multi_peft

# Command Line Arguments
parser = argparse.ArgumentParser(description='Main program')
parser.add_argument('--base_model', type=str, help='Path to or name of base model')
parser.add_argument('--model_type', type=str, default="llama", help='The model type, support: llama, chatglm')
parser.add_argument('--inference', action="store_true", help='The inference mode (just for test)')
parser.add_argument('--load_lora', action="store_true", help="Load lora from file instead of init randomly")
parser.add_argument('--disable_lora', action="store_true", help="Disable the lora modules")
parser.add_argument('--tokenizer', type=str, help='Path to or name of tokenizer')
parser.add_argument('--load_8bit', action="store_true", help='Load model in 8bit mode')
parser.add_argument('--load_4bit', action="store_true", help='Load model in 4bit mode')
parser.add_argument('--device', type=str, default='cuda:0', help='Specify which GPU to be used, default is cuda:0')
parser.add_argument('--config', type=str, help='Path to finetune configuration')
parser.add_argument('--seed', type=int, default=42, help='Random seed in integer, default is 42')
parser.add_argument('--log', type=bool, default=True, help='Turn on or off log, default is true')

args = parser.parse_args()


def log(msg: str):
    if args.log:
        print('[%s] %s' %
              (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))


if torch.cuda.is_available():
    log('NVIDIA CUDA initialized successfully.')
    log('Total %i GPU(s) detected.' % torch.cuda.device_count())
else:
    print('NVIDIA CUDA computing capacity required. Please check your PyTorch installation.')
    exit(-1)

if args.base_model is None:
    print('error: Argument --base_model are required.')
    parser.print_help()
    exit(-1)

if args.config is None:
    print('error: Argument --config are required.')
    parser.print_help()
    exit(-1)


# Functions
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def load_base_model() -> Tuple[multi_peft.Tokenizer, multi_peft.BaseModelMixin]:
    if args.model_type == 'llama':
        model = multi_peft.LlamaModel.from_pretrained(
            path=args.base_model,
            model_type=args.model_type,
            device=args.device,
            bits=(8 if args.load_8bit else (4 if args.load_4bit else None)),
            log_fn=log,
        )
        tokenizer = multi_peft.Tokenizer(args.base_model)
        model.pad_token_id = tokenizer.pad_id
        return tokenizer, model

    if args.model_type == 'opt':
        model = multi_peft.OPTModel.from_pretrained(
            path=args.base_model,
            model_type=args.model_type,
            device=args.device,
            bits=(8 if args.load_8bit else (4 if args.load_4bit else None)),
            log_fn=log,
        )
        tokenizer = multi_peft.Tokenizer(args.base_model)
        model.pad_token_id = tokenizer.pad_id
        return tokenizer, model

    raise ValueError('Invalid model')


def init_lora_model(config: Dict[str, any], model: multi_peft.BaseModelMixin):
    if args.disable_lora:
        return

    for lora_config in config["lora"]:
        lora_weight = None
        if args.load_lora:
            adapter_file_path = lora_config["output"] + "/adapter_model.bin"
            print(f"load {adapter_file_path}")
            lora_weight = torch.load(adapter_file_path)

        model.init_lora_weight(
            lora_config["agent_id"],
            lora_config["r"],
            lora_config["alpha"],
            lora_config["dropout"],
            lora_config["target_modules"],
            lora_weight,
        )


def get_optimizer(config: Dict[str, any], train_params: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.optim.Optimizer]:
    # get optimizer per lora model
    optimizer: Dict[str, torch.optim.Optimizer] = {}

    for lora_config in config["lora"]:
        agent_id = lora_config["agent_id"]
        optim_name = lora_config["optim"]
        lr = lora_config["lr"]
        if optim_name == "sgd":
            momentum = 0
            if "momentum" in lora_config:
                momentum = lora_config["momentum"]
            optimizer[agent_id] = (torch.optim.SGD(train_params[agent_id], lr=lr, momentum=momentum))
        elif optim_name == "adamw":
            if train_params.get(agent_id) is None:
                print(f"no params for {agent_id}")
            optimizer[agent_id] = (torch.optim.AdamW(train_params[agent_id], lr=lr))
        else:
            raise f"unkown optimizer {optim_name}"

    return optimizer


def get_accumulation_steps(config: Dict[str, any]) -> Dict[str, int]:
    ret_accumulation_step = {}
    for lora_config in config["lora"]:
        batch_size = lora_config["batch_size"]
        micro_batch_size = lora_config["micro_batch_size"]
        if batch_size < micro_batch_size or batch_size % micro_batch_size != 0:
            raise f"error batch_size {batch_size} and micro batch size {micro_batch_size}"
        ret_accumulation_step[lora_config["agent_id"]] = batch_size / micro_batch_size
    return ret_accumulation_step


# to get test result and want early stop it
def train(config: Dict[str, any], llm_model: multi_peft.BaseModelMixin, dispatcher: multi_peft.Dispatcher):
    # the train params per lora model
    all_train_params: Dict[str, List[torch.Tensor]] = llm_model.get_train_params(config)
    all_optimizer: Dict[str, torch.optim.Optimizer] = get_optimizer(config, all_train_params)
    accumulation_step: Dict[str, int] = get_accumulation_steps(config)
    loss_fn = torch.nn.CrossEntropyLoss()

    step_cnt = 0
    log(f'Total {len(dispatcher.lora_agents)} task(s)')
    while not len(dispatcher.lora_agents) > 0:
        step_cnt += 1

        inputs: multi_peft.MultiBatchInputConfig = dispatcher.get_train_data()
        for lora in inputs.batch_input_config_list:
            all_optimizer[lora.agent_id].zero_grad()

        output = llm_model.forward(inputs)
        labels = torch.tensor(inputs.tokens, dtype=torch.long).to(args.device)

        total_loss = None
        idx = 0
        for lora_config in inputs.batch_input_config_list:
            start_idx = idx
            end_idx = idx + lora_config.batch_size
            idx = end_idx

            loss_input = output[start_idx:end_idx][..., :-1, :].contiguous().view(-1, llm_model.vocab_size)
            loss_target = labels[start_idx:end_idx][..., 1:].contiguous().view(-1)
            loss = loss_fn(loss_input, loss_target) / accumulation_step[lora_config.agent_id]

            print(f"    adapter: {lora_config.agent_id} loss: {loss}")

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        total_loss.backward()

        for lora in inputs.batch_input_config_list:
            if step_cnt % accumulation_step[lora.agent_id] == 0:
                all_optimizer[lora.agent_id].step()

        if step_cnt % config["save_step"] == 0:
            multi_peft.save_lora_model(llm_model, config, f"{step_cnt}")

    multi_peft.save_lora_model(llm_model, config)


def inference(config: Dict[str, any], llm_model: multi_peft.BaseModelMixin, tokenizer: multi_peft.Tokenizer):
    lora_adapter_num = len(config["lora"])
    batch_data_config: List[multi_peft.BatchInputConfig] = []

    for idx, lora_config in enumerate(config["lora"]):
        agent_id = lora_config["agent_id"]
        batch_data_config.append(multi_peft.BatchInputConfig(agent_id, 1))

    inference_max_len = 128

    while True:
        input_raw = input("INPUT WITHOUT PROMPT: ")
        if input_raw == "QUIT":
            return

        tokens = tokenizer.encode(input_raw, True, False)
        token_len = len(tokens)
        while len(tokens) < inference_max_len:
            tokens.append(tokenizer.pad_id)

        input_data = multi_peft.MultiBatchInputConfig(
            prompts=[input_raw] * lora_adapter_num,
            batch_input_config_list=batch_data_config,
            tokens=[tokens] * lora_adapter_num,
            tokens_len_without_pad=[token_len] * lora_adapter_num,
            batch_seq_len=inference_max_len,
            is_inference=True,
        )

        eos_flag: List[bool] = [False] * lora_adapter_num
        for pos in range(token_len, inference_max_len):
            with torch.no_grad():
                # batch_size, seq_len, voc_logs
                outputs = llm_model.forward(input_data)
                next_token = outputs[:, pos - 1, :]
                next_token = torch.argmax(next_token, dim=-1)
                for idx in range(len(input_data.tokens)):
                    input_data.tokens[idx][pos] = next_token[idx].item()
                    # end of the sentence
                    if next_token[idx].item() == tokenizer.eos_id:
                        eos_flag[idx] = True
                    input_data.tokens_len_without_pad[
                        idx] = input_data.tokens_len_without_pad[idx] + 1
            # check if the all sentence end
            have_all_done = all(flag for flag in eos_flag)
            if have_all_done:
                break

        for idx, output in enumerate(input_data.tokens):
            print(f"# LORA{idx} OUTPUT IS:")
            print(tokenizer.decode(output))


class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls to GPUtil/sec
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


if __name__ == "__main__":
    setup_seed(args.seed)

    with open(args.config, 'r', encoding='utf8') as fp:
        config = json.load(fp)

    tokenizer, model = load_base_model()
    init_lora_model(config, model)

    torch.cuda.empty_cache()

    monitor = Monitor(1)
    start_time = time.time()

    if args.inference:
        inference(config, model, tokenizer)
    else:
        dispatcher = multi_peft.Dispatcher(config, tokenizer)
        max_gpu_usage = train(config, model, dispatcher)

    end_time = time.time()
    monitor.stop()

    total_time = end_time - start_time
    print(f"Total Running Time: {total_time} seconds")
