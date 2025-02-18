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
import numpy as np

def get_folders(root_dir: str) -> list:
    root = Path(root_dir)
    return [f for f in root.iterdir() if f.is_dir()]

def get_base_name(folder_name: str) -> Tuple[str, int]:
    """从文件夹名称中提取基础名称（不包含random_seed）和seed值"""
    if "random_seed" not in folder_name:
        return folder_name, None
    
    parts = folder_name.split("--")
    seed = None
    base_parts = []
    
    for part in parts:
        if part.startswith("random_seed"):
            seed = int(part.split("(")[1].rstrip(")"))
        else:
            base_parts.append(part)
    
    return "--".join(base_parts), seed

def get_accuracy_from_metrics(metric_data: Dict) -> Tuple[float, float]:
    """从metrics数据中提取准确率和标准差"""
    if 'accuracy' in metric_data:
        # 单次实验
        return metric_data['accuracy'] * 100, 0.0
    elif 'avg_accuracy' in metric_data and 'std_accuracy' in metric_data:
        # 多次实验
        return metric_data['avg_accuracy'] * 100, metric_data['std_accuracy'] * 100
    else:
        raise KeyError("No accuracy information found in metrics")

def main(model_name: str):
    root_dir = Path("cache/log")
    folders = get_folders(root_dir/model_name)

    # 使用嵌套的defaultdict来存储结果
    results = defaultdict(lambda: defaultdict(list))
    final_results = {}

    # 收集数据
    for folder in folders:
        try:
            with open(folder/"metrics.json", 'r') as f:
                metric_data = json.load(f)
        except FileNotFoundError:
            print("File not found, skipping", folder)
            continue
        
        try:
            if isinstance(metric_data, list):
                metric_data = metric_data[0]  # 如果是列表，取第一个元素
            acc, std = get_accuracy_from_metrics(metric_data)
        except KeyError as e:
            print(f"Error processing {folder}: {e}")
            continue
        
        base_name, seed = get_base_name(folder.name)
        
        if seed is not None:
            # 存储带有seed的结果
            results[base_name]['accuracies'].append(acc)
            results[base_name]['stds'].append(std)
        else:
            # 存储单个结果
            final_results[folder.name] = (acc, std)

    # 处理多seed实验结果
    for base_name, data in results.items():
        accuracies = np.array(data['accuracies'])
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        final_results[base_name] = (mean_acc, std_acc)

    # 按准确率排序并打印所有结果
    for name, (acc, std) in sorted(final_results.items(), key=lambda x: x[1][0], reverse=True):
        print(f"{name}: {acc:.1f}±{std:.1f}")

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dir", type=str, default="Meta-Llama-3-8B-Instruct-concat/ablation/selected_obf_text(all)/obf_technique_test")
    args = args.parse_args()
    main(args.dir)
