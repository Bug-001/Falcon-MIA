import os
import yaml
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple, Any
import pickle
from collections import defaultdict
import argparse
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

def get_folders(root_dir: str) -> list:
    root = Path(root_dir)
    return [f for f in root.iterdir() if f.is_dir()]

def main():
    root_dir = Path("cache/log")
    model_name = 'Meta-Llama-3-8B-Instruct/obf_technique_test'
    folders = get_folders(root_dir/model_name)

    l = {}

    # 收集数据
    for folder in folders:
        try:
            with open(folder/"metrics.json", 'r') as f:
                metric_data = json.load(f)
        except FileNotFoundError:
            print("File not found, skipping", folder)
            continue
        acc = metric_data[0]['accuracy']
        l[folder.name] = acc

    ans = sorted(l.items(), key=lambda x: x[1], reverse=True)
    for k, v in ans:
        print(k, v)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    main()