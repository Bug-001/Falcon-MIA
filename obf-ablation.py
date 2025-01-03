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

import obf_multi_techniques, obf_metrics_getter
from obf_multi_techniques import parse_folder_name, get_folder_name

def get_folders(root_dir: str) -> list:
    root = Path(root_dir)
    return [f for f in root.iterdir() if f.is_dir()]

def organize_folders(root_dir: str, result_dir: str, ablation_vars: List[str]):
    """
    整理文件夹，将符合命名规范的文件夹移动到ablation目录下的子文件夹中
    """
    root = Path(root_dir)
    model_dir = root / result_dir
    ablation_dir = model_dir / "ablation"

    if not ablation_dir.exists():
        ablation_dir.mkdir(parents=True, exist_ok=True)
    
    # 遍历文件夹
    for folder in get_folders(model_dir):
        info = parse_folder_name(folder.name)
        if not info:
            continue
        
        # 提取并排序ablation变量的值
        ablation_values = [(var, info[var]) for var in sorted(ablation_vars) if var in info]
        if not ablation_values:
            continue
        
        # 创建子文件夹名
        folder_name = "--".join(f"{var}({value})" for var, value in ablation_values)
        target_dir = ablation_dir / folder_name
        os.makedirs(target_dir, exist_ok=True)
        
        # 复制文件夹
        for var in ablation_vars:
            info.pop(var, None)
        info.pop('type', None) # XXX
        new_folder = target_dir / get_folder_name(info)
        if not new_folder.exists():
            shutil.copytree(folder, new_folder)

def main(data_config, attack_config, query_config, ablation_vars: List[str], result_dir: str):
    print(ablation_vars, result_dir)
    root_dir = Path("cache/log")
    
    # 整理文件夹
    organize_folders(root_dir, result_dir, ablation_vars)
    
    # 对ablation目录下的每个子文件夹进行处理
    ablation_dir = root_dir / result_dir / "ablation"
    for sub_dir in get_folders(ablation_dir):
        
        print(f"Processing {sub_dir.name}...")
        # 直接调用obf-multi-techniques的main函数
        obf_multi_techniques.main(data_config, attack_config, query_config, model_name=f"{result_dir}/ablation/{sub_dir.name}")

    # 处理完成后，使用metrics_getter整合所有结果
    for sub_dir in get_folders(ablation_dir):
        print(f"Showing the metrics of {sub_dir.name}...")
        obf_metrics_getter.main(model_name=f"{result_dir}/ablation/{sub_dir.name}/obf_technique_test")

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
    parser.add_argument("--ablation-vars", nargs='+', help="List of variable names for ablation", required=True)
    parser.add_argument("--result-dir", help="Directory name for ablation results", required=True)
    args = parser.parse_args()

    data_config = load_yaml_config(args.data)
    attack_config = load_yaml_config(args.attack)
    query_config = load_yaml_config(args.query)
    main(data_config, attack_config, query_config, args.ablation_vars, args.result_dir)
