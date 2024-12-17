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
import shutil

from attack.obfuscation import ObfuscationAttack
import icl

import obf_multi_techniques
from obf_multi_techniques import parse_folder_name, get_folder_name

def get_folders(root_dir: str) -> list:
    root = Path(root_dir)
    return [f for f in root.iterdir() if f.is_dir()]

def organize_folders(root_dir: str, model_name: str):
    """
    整理文件夹，将符合命名规范的文件夹移动到ablation目录下的四个子文件夹中
    """
    root = Path(root_dir)
    model_dir = root/model_name
    ablation_dir = model_dir/"ablation"

    if not ablation_dir.exists():
        ablation_dir.mkdir(parents=True, exist_ok=True)
    else:
        return
    
    # 创建四个子文件夹
    sub_dirs = {
        (True, True): ablation_dir/"obf_idf_sim_idf",
        (True, False): ablation_dir/"obf_idf_sim_no_idf",
        (False, True): ablation_dir/"obf_no_idf_sim_idf",
        (False, False): ablation_dir/"obf_no_idf_sim_no_idf"
    }
    
    # 创建文件夹
    for dir_path in sub_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # 移动文件
    for folder in get_folders(model_dir):
        info = parse_folder_name(folder.name)
        if not info:
            continue
        
        target_dir = sub_dirs[(info['obf_use_idf'], info['sim_use_idf'])]
        
        # 创建新的文件夹名
        info_copy = info.copy()
        info_copy.pop('sim_use_idf', None)
        info_copy.pop('obf_use_idf', None)
        info_copy.pop('type', None)
        new_folder = target_dir/get_folder_name(info_copy)
        if new_folder.exists():
            continue
            
        # 复制文件夹
        shutil.copytree(folder, new_folder)

def main(data_config, attack_config, query_config, model_name='Meta-Llama-3-8B-Instruct-ablation'):
    root_dir = Path("cache/log")
    
    # 整理文件夹
    organize_folders(root_dir, model_name)
    
    # 对ablation目录下的每个子文件夹进行处理
    ablation_dir = root_dir/model_name/"ablation"
    for sub_dir in get_folders(ablation_dir):
        
        print(f"Processing {sub_dir.name}...")
        # 直接调用obf-multi-techniques的main函数
        obf_multi_techniques.main(data_config, attack_config, query_config, model_name=f"{model_name}/ablation/{sub_dir.name}")

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    # Keep the original command-line argument parsing for backward compatibility
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='Path to the data config file', default="data.yaml")
    parser.add_argument('--attack', help='Path to the attack config file', default="attack_chat.yaml")
    parser.add_argument('--query', help='Path to the query config file', default="query.yaml")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help="Set the logging level")
    parser.add_argument("--output", help="Output directory for all results", default=None)
    args = parser.parse_args()

    data_config = load_yaml_config(args.data)
    attack_config = load_yaml_config(args.attack)
    query_config = load_yaml_config(args.query)
    main(data_config, attack_config, query_config)
