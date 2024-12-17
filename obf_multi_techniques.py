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

from attack.obfuscation import ObfuscationAttack
import icl

def parse_folder_name(folder_name: str) -> Dict[str, Any]:
    result = {}
    parts = folder_name.split('--')
    
    for part in parts:
        if '(' not in part or ')' not in part:
            continue
            
        key = part[:part.find('(')]
        value = part[part.find('(')+1:part.find(')')]
        
        if key == 'num_demonstrations':
            value = int(value)
        elif key in ['sim_use_idf', 'obf_use_idf']:
            value = value.lower() == 'true'
            
        result[key] = value
        
    return result

def get_folder_name(info: Dict[str, str]) -> str:
    parts = []
    
    # 优先添加task和dataset
    if 'dataset' in info:
        parts.append(f"dataset({info['dataset']})")
    if 'task' in info:
        parts.append(f"task({info['task']})")
        
    # 添加其他键值对
    other_parts = []
    for key, value in sorted(info.items()):
        if key not in ['task', 'dataset']:
            other_parts.append(f"{key}({value})")
            
    parts.extend(other_parts)
    
    return '--'.join(parts)

def load_similarities(folder_path: Path) -> List[Tuple[Dict, Dict]]:
    sim_path = folder_path / "similarities_data"
    if not sim_path.exists():
        return []
    with open(sim_path, 'rb') as f:
        return pickle.load(f)

def get_folders(root_dir: str) -> list:
    root = Path(root_dir)
    return [f for f in root.iterdir() if f.is_dir()]

def main(data_config, attack_config, query_config, model_name='Meta-Llama-3-8B-Instruct'):
    root_dir = Path("cache/log")
    folders = get_folders(root_dir/model_name)
    
    # 收集数据
    raw_data = []
    for folder in folders:
        info = parse_folder_name(folder.name)
        if not info:
            continue
            
        similarities = load_similarities(folder)
        if not similarities:
            continue
        
        group_key = (info['task'], info['dataset'], info['num_demonstrations'])
        technique = info['technique'][:3]
        
        # 处理相似度名称
        processed_similarities = []
        for data, label in similarities:
            new_data = {}
            for level, sim_dict in data.items():
                new_sim_dict = {}
                for sim_name, sim_value in sim_dict.items():
                    new_name = f"{technique}_{sim_name[:3]}"
                    new_sim_dict[new_name] = sim_value
                new_data[level] = new_sim_dict
            processed_similarities.append((new_data, label))
            
        raw_data.append({
            'key': group_key,
            'data': processed_similarities
        })
    
    # 按key分组合并数据
    grouped_data = defaultdict(list)
    for item in raw_data:
        grouped_data[item['key']].append(item['data'])
    
    # 合并相同位置的数据
    final_results = {}
    for key, data_list in grouped_data.items():
        merged_data = []
        data_length = len(data_list[0])
        
        for i in range(data_length):
            new_sim_dict = {}
            for data in data_list:
                current_similarities, label = data[i]
                for level, sim_dict in current_similarities.items():
                    if level not in new_sim_dict:
                        new_sim_dict[level] = {}
                    for sim_name, sim_value in sim_dict.items():
                        new_sim_dict[level][sim_name] = sim_value
            # Assert all values of new_sim_dict has the same length
            for level, sim_dict in new_sim_dict.items():
                assert len(sim_dict) == len(new_sim_dict[next(iter(new_sim_dict))]) 
            merged_data.append((new_sim_dict, label))
            
        final_results[key] = merged_data
    
    # 借用ObfuscationAttack的方法进行分析
    for key, similarities_data in final_results.items():
        data_config = {
            "dataset": key[1],
            "task": key[0],
            "num_demonstrations": key[2]
        }
        data_config['technique'] = 'character_swap' # 假设该文件夹已经准备好
        attack_config['type'] = 'Obfuscation'
        attack_config['technique'] = 'obf_technique_test'
        attack_config['name'] = f'{model_name}/obf_technique_test/{key}'
        attack_config['num_similarities'] = len(list(similarities_data[0][0].values())[0])
        attack_config['train_attack'] = 400
        attack_config['test_attack'] = 100
        attack_config['attack_phase'] = 'train-test'
        # 将similarities_data保存，从而攻击会跳过访问LLM的阶段
        os.makedirs(root_dir/attack_config['name'], exist_ok=True)
        with open(root_dir/attack_config['name']/"similarities_data", 'wb') as f:
            pickle.dump(similarities_data, f)
        # 从对应的文件夹中加载一个dataset_overview.json文件，保存起来，以便attack.evaluate访问
        # XXX: 以后应该把数据统一放到一个文件夹中
        source_path = root_dir/model_name/get_folder_name(data_config)/"dataset_overview.json"
        dataset_overview_path = Path(root_dir/attack_config['name']/"dataset_overview.json")
        with open(source_path, 'r') as f:
            dataset_overview = json.load(f)
            with open(dataset_overview_path, 'w') as g:
                json.dump(dataset_overview, g, indent=4)
        # 创建一个假的level_info文件，这里就不再获取真的level_info了
        level_info_path = Path(root_dir/attack_config['name']/"level_details.json")
        with open(level_info_path, 'w') as f:
            json.dump([{
                "sample_id": -1,
                "response": "",
            }], f, indent=4)
        print('Starting attack:', key)
        icl.main(data_config, attack_config, query_config)

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