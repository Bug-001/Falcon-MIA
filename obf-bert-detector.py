import os
import yaml
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple, Any
import pickle
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
from transformers import Trainer, TrainingArguments, LongformerForSequenceClassification, LongformerTokenizer
from collections import defaultdict
from dataclasses import dataclass
import argparse
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax

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

def get_folders(root_dir: str) -> list:
    root = Path(root_dir)
    return [f for f in root.iterdir() if f.is_dir()]

@dataclass
class ObfuscationDetectorCollator:
    tokenizer: LongformerTokenizer
    max_length: int = 1024
    
    def __call__(self, features: List[Tuple[Dict[float, str], bool]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)

        # Let's not introduce levels yet... just put sentences in a row
        labels = []

        all_responses = []
        # Tokenize data, put all responses together
        for responses_dict, label in features:
            labels.append(label)
            # Original text is at level 0
            responses = []
            for k in sorted(responses_dict):
                response = responses_dict[k]
                response = response[:len(response)*4//5]
                responses.append(response)
            all_responses.append(self.tokenizer.sep_token.join(responses))

        # tokenize to: [CLS] response1 [SEP] response2 [SEP] response3 [SEP] ...
        tokenized_responses = self.tokenizer(all_responses, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            "input_ids": tokenized_responses['input_ids'],
            "attention_mask": tokenized_responses['attention_mask'],
            "labels": torch.tensor(labels, dtype=torch.float32),
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (logits > 0.5).astype(np.int64)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def main(data_config, attack_config, query_config, name='unnamed_experiment'):
    root_dir = Path("cache/log")
    model_name = 'Meta-Llama-3-8B-Instruct-truncated'
    detector_name = "allenai/longformer-base-4096"
    folders = get_folders(root_dir/model_name)

    tokenizer = LongformerTokenizer.from_pretrained(detector_name, do_lower_case=True)
    collator = ObfuscationDetectorCollator(tokenizer)

    # Collect data
    for folder in folders:
        info = parse_folder_name(folder.name)
        if not info:
            continue

        # Check if already trained
        if Path(f"cache/model/detector/{model_name}/{folder.name}/model.safetensors").exists():
            continue

        # Ensure level_details.json exists
        level_details_path = folder/"level_details.json"
        if not level_details_path.exists():
            continue

        if folder.name != "task(classification)--dataset(cce)--num_demonstrations(6)--technique(leet_speak)":
            continue

        model = LongformerForSequenceClassification.from_pretrained(detector_name, num_labels=1)
        model.cuda()

        dataset_overview = pd.read_json(folder/"dataset_overview.json")
        dataset_overview.set_index('sample_id', inplace=True)
        level_details = pd.read_json(folder/"level_details.json")
        responses = []
        # Iterate through dataframe data, first group by sample_id
        for sample_id, group in tqdm(level_details.groupby('sample_id'), desc=folder.name):
            all_level_responses = dict()
            orig_data = dataset_overview.iloc[sample_id]
            for index, row in group.iterrows():
                original = orig_data['Original']
                response = row['response']
                all_level_responses[row['level']] = response
                all_level_responses[0] = original
            responses.append((all_level_responses, orig_data['Membership']))

        # Set training parameters
        training_args = TrainingArguments(
            output_dir=f"cache/model/detector/{model_name}/{folder.name}",
            eval_strategy="steps",
            eval_steps=20,
            save_strategy="steps",
            save_steps=20,
            save_total_limit=10,
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=30,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            # remove_unused_columns=False,
            report_to="wandb",          # Enable wandb logging
            logging_dir="cache/model/detector/log",
            logging_steps=5,
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=responses[:400],
            eval_dataset=responses[400:],
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        trainer.save_model(f"cache/model/detector/{model_name}/{folder.name}")

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