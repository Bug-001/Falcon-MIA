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

def main(model_name: str):
    root_dir = Path("cache/log")
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
        acc = metric_data[0]['avg_accuracy'] * 100
        std = metric_data[0]['std_accuracy'] * 100
        l[folder.name] = (acc, std)

    ans = sorted(l.items(), key=lambda x: x[1], reverse=True)
    for k, v in ans:
        print(k, f"{v[0]:.1f}±{v[1]:.1f}")


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dir", type=str, default="Meta-Llama-3-8B-Instruct-concat/ablation/selected_obf_text(all)/obf_technique_test")
    args = args.parse_args()
    main(args.dir)
