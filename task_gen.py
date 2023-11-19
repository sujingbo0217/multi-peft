# @Time : 11/3/23 10:49 AM
# @Author : Jingbo Su
# @File : task_gen.py
# @Software : PyCharm

import argparse
import json
import random


def generate_lora_config(agent_id: str, model_type: str):
    config = {
        "agent_id": agent_id,
        "output": agent_id,
        "optim": "adamw",
        "lr": round(random.uniform(1e-4, 1e-3), 4),
        "batch_size": random.choice([32, 64, 128]),
        "micro_batch_size": random.choice([16, 32, 64]),
        "test_batch_size": 64,
        "num_epochs": random.choice([10, 20, 30]),
        "r": random.choice([4, 8]),
        "alpha": 16,
        "dropout": round(random.uniform(0.01, 0.1), 2),
        "target_modules": {
            "q_proj": random.choice([True, False]),
            "k_proj": False,
            "v_proj": random.choice([True, False]),
        },
        "data": random.choice(["math_10k.json", "math_50k.json", "commonsense_15k.json", "commonsense_170k.json"])
    }
    while config.get('batch_size') < config.get('micro_batch_size'):
        config['batch_size'] = random.choice([32, 64, 128])
        config['micro_batch_size'] = random.choice([16, 32, 64])
    if model_type == 'llama':
        config['target_modules']['o_proj'] = False
        config['target_modules']['w1_proj'] = False
        config['target_modules']['w2_proj'] = False
        config['target_modules']['w3_proj'] = False
    elif model_type == 'opt':
        config['target_modules']['output_proj'] = False
        config['target_modules']['fc1'] = False
        config['target_modules']['fc2'] = False
    else:
        pass
    while not (config.get('target_modules').get('q_proj') or config.get('target_modules').get('v_proj')):
        config['target_modules']['q_proj'] = random.choice([True, False])
        config['target_modules']['v_proj'] = random.choice([True, False])
    return config


parser = argparse.ArgumentParser(description='Get Agent Number')
parser.add_argument('--num_agents', type=int, help='Agent number')
parser.add_argument('--model_type', type=str, help='Model type')
args = parser.parse_args()

num_agents = args.num_agents
model_type = args.model_type

lora_configurations = {
    "cutoff_len": 256,
    "group_by_length": False,
    "expand_right": True,
    "pad_token_id": -1,
    "save_step": 2000,
    "early_stop_test_step": 2000,
    "lora": [generate_lora_config(f"agent_{i}", model_type) for i in range(num_agents)]
}

with open("data_max.json", "w") as json_file:
    json.dump(lora_configurations, json_file, indent=2)
