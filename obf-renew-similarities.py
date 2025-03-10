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

from attack.obfuscation import ObfuscationAttack
from string_utils import StringHelper
from utils import SDatasetManager
import icl

# This script will use generated LLM response to conduct some tiny experiments.

def parse_folder_name(folder_name: str) -> Dict[str, str]:
    """Parse path in format task(xx)--dataset(xx)--num_demonstrations(xx)--technique(xx)"""
    pattern = r"task\((\w+)\)--dataset\((\w+)\)--num_demonstrations\((\d+)\)--technique\((\w+)\)"
    match = re.match(pattern, folder_name)
    if not match:
        return {}
    return {
        "task": match.group(1),
        "dataset": match.group(2),
        "num_demonstrations": int(match.group(3)),
        "technique": match.group(4)
    }

def get_folder_name(info: Dict[str, str]) -> str:
    return f"task({info['task']})--dataset({info['dataset']})--num_demonstrations({info['num_demonstrations']})--technique({info['technique']})"

def load_similarities(folder_path: Path) -> List[Tuple[Dict, Dict]]:
    sim_path = folder_path / "similarities_data"
    if not sim_path.exists():
        return []
    with open(sim_path, 'rb') as f:
        return pickle.load(f)
    
def save_similarities(folder_path: Path, similarities: List[Tuple[Dict, Dict]]):
    sim_path = folder_path / "similarities_data"
    with open(sim_path, 'wb') as f:
        pickle.dump(similarities, f)

def get_folders(root_dir: str) -> list:
    root = Path(root_dir)
    return [f for f in root.iterdir() if f.is_dir()]

def renew_similarity(folder_path: Path, info: Dict):
    # if (folder_path/"similarities_data_processed.txt").exists():
    #     return load_similarities(folder_path)

    sdm = SDatasetManager(info['dataset'], info['task'])
    shelper = StringHelper()

    # Enable IDF here
    shelper.set_idf_dict(sdm.get_idf())
    
    similarities_data = []
    try:
        level_details = pd.read_json(folder_path/"level_details.json")
        dataset_overview = pd.read_json(folder_path/"dataset_overview.json")
        dataset_overview.set_index('sample_id', inplace=True)
    except FileNotFoundError:
        print("File not found, skipping", folder_path)
        return []

    new_level_details = level_details.copy()
    
    # Iterate through dataframe data, first group by sample_id
    for sample_id, group in tqdm(level_details.groupby('sample_id'), desc=folder_path.name):
        all_level_similarities = dict()
        orig_data = dataset_overview.iloc[sample_id]
        for index, row in group.iterrows():
            original = orig_data['Original']
            response = row['response'].split('\n')[0]
            similarities = shelper.calculate_overall_similarity_dict(original, response)
            new_level_details.at[index, 'similarities'] = similarities
            all_level_similarities[row['level']] = similarities
        similarities_data.append((all_level_similarities, orig_data['Membership']))

    new_level_details.to_json(folder_path/"level_details.json", indent=4, orient='records')
    save_similarities(folder_path, similarities_data)
    with open(folder_path/"similarities_data_processed.txt", 'w') as f:
        f.write("Done!")

    return similarities_data

def main(data_config, attack_config, query_config, name='unnamed_experiment'):
    root_dir = Path("cache/log")
    model_name = 'Meta-Llama-3-8B-Instruct'
    folders = get_folders(root_dir/model_name)

    shelper = StringHelper()
    
    l = {}

    # Collect data
    for folder in folders:
        info = parse_folder_name(folder.name)
        if not info:
            continue

        # if folder.name != "task(platform_detection)--dataset(cce)--num_demonstrations(6)--technique(leet_speak)":
        #     continue

        if not folder.name.startswith("task(platform_detection)--dataset(cce)--num_demonstrations(6)"):
            continue

        similarities_data = renew_similarity(folder, info)

        continue
        
        # Call icl function to start model training
        data_config = {
            "dataset": info['dataset'],
            "task": info['task'],
            "num_demonstrations": info['num_demonstrations'],
        }
        attack_config['type'] = 'Obfuscation'
        attack_config['technique'] = info['technique']
        attack_config['name'] = f"{model_name}/{folder.name}"
        attack_config['num_similarities'] = len(list(similarities_data[0][0].values())[0])
        print('Starting attack:', folder.name)
        icl.main(data_config, attack_config, query_config)

    #     with open(folder/"metrics.json", 'r') as f:
    #         metric_data = json.load(f)
    #     acc = metric_data[0]['accuracy']
    #     l[folder.name] = acc

    # ans = sorted(l.items(), key=lambda x: x[1], reverse=True)
    # for k, v in ans:
    #     print(k, v)


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