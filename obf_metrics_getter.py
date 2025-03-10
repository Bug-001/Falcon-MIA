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
    """Extract base name (without random_seed) and seed value from folder name"""
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
    """Extract accuracy and standard deviation from metrics data"""
    if 'accuracy' in metric_data:
        # Single experiment
        return metric_data['accuracy'] * 100, 0.0
    elif 'avg_accuracy' in metric_data and 'std_accuracy' in metric_data:
        # Multiple experiments
        return metric_data['avg_accuracy'] * 100, metric_data['std_accuracy'] * 100
    else:
        raise KeyError("No accuracy information found in metrics")

def main(model_name: str):
    root_dir = Path("cache/log")
    folders = get_folders(root_dir/model_name)

    # Use nested defaultdict to store results
    results = defaultdict(lambda: defaultdict(list))
    final_results = {}

    # Collect data
    for folder in folders:
        try:
            with open(folder/"metrics.json", 'r') as f:
                metric_data = json.load(f)
        except FileNotFoundError:
            print("File not found, skipping", folder)
            continue
        
        try:
            if isinstance(metric_data, list):
                metric_data = metric_data[0]  # If it's a list, take the first element
            acc, std = get_accuracy_from_metrics(metric_data)
        except KeyError as e:
            print(f"Error processing {folder}: {e}")
            continue
        
        base_name, seed = get_base_name(folder.name)
        
        if seed is not None:
            # Store results with seed
            results[base_name]['accuracies'].append(acc)
            results[base_name]['stds'].append(std)
        else:
            # Store single result
            final_results[folder.name] = (acc, std)

    # Process multi-seed experiment results
    for base_name, data in results.items():
        accuracies = np.array(data['accuracies'])
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        final_results[base_name] = (mean_acc, std_acc)

    # Sort by accuracy and print all results
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
