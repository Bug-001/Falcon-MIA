# icl_attack.py

import yaml
import argparse
import random
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from llm.tools.utils import get_logger
from llm.query import QueryProcessor

class ModelInterface:
    def __init__(self, query_config):
        self.query_config = query_config
        
    def query(self, prompt: List[Dict[str, str]], chat_name: str) -> str:
        config = self.query_config.copy()
        chat_config = {
            "name": chat_name,
            "messages": prompt
        }
        config['chats'] = [chat_config]
        
        llm_response = QueryProcessor(config).process_query()[0]
        return llm_response

class ICLDataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True, seed: int = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed if seed is not None else random.randint(0, 1000000)
        self._iterator = None

    def __iter__(self):
        self._iterator = iter(self.dataset.shuffle(seed=self.seed).iter(batch_size=self.batch_size))
        return self

    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration and TypeError:
            self._iterator = iter(self.dataset.shuffle(seed=self.seed).iter(batch_size=self.batch_size))
            return next(self._iterator)

class ICLAttackStrategy(ABC):
    def __init__(self, attack_config: Dict[str, Any]):
        self.attack_config = attack_config
        self.random_seed = attack_config.get('random_seed', random.randint(0, 1000000))
        random.seed(self.random_seed)
        self.results = []
        self.label_translation = {}

    def prepare(self, data_config: Dict[str, Any]):
        self.data_config = data_config
        self.dataset = load_dataset(data_config['name'])
        self.input_field = data_config['input_field']
        self.output_field = data_config['output_field']
        self.label_translation = data_config.get('label_translation', {})
        
        batch_size = data_config.get('num_demonstrations', 1)
        self.train_loader = ICLDataLoader(self.dataset['train'], batch_size=batch_size, seed=self.random_seed)
        self.test_loader = ICLDataLoader(self.dataset['test'], batch_size=1, seed=self.random_seed)

    def translate_label(self, label):
        return self.label_translation.get(label, label)

    def get_demo_template(self):
        return self.data_config['icl_prompt']['demonstration_template']

    def generate_icl_prompt(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        prompt = self.data_config['icl_prompt']['initial_conversation'].copy()
        demonstration_template = self.data_config['icl_prompt']['demonstration_template']

        batch = next(self.train_loader)
        icl_samples = []

        for i in range(len(batch[self.input_field])):
            sample = {
                "input": batch[self.input_field][i],
                "output": self.translate_label(batch[self.output_field][i])
            }
            icl_samples.append(sample)
            for item in demonstration_template:
                prompt.append({
                    "role": item['role'],
                    "content": item['content'].format(**sample)
                })

        return icl_samples, prompt

    def get_attack_sample(self, icl_samples) -> Tuple[Dict[str, Any], bool]:
        if random.random() < 0.5 and icl_samples:
            sample = random.choice(icl_samples)
            is_member = True
        else:
            batch = next(self.train_loader if random.random() < 0.5 else self.test_loader)
            index = random.randint(0, len(batch[self.input_field]) - 1)
            sample = {
                "input": batch[self.input_field][index],
                "output": self.translate_label(batch[self.output_field][index])
            }
            is_member = False
        
        return sample, is_member

    @abstractmethod
    def attack(self, model: 'ModelInterface'):
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        pass

    @staticmethod
    def create(attack_config: Dict[str, Any]) -> 'ICLAttackStrategy':
        attack_type = attack_config['type']
        if attack_type == 'GAP':
            return GAPAttack(attack_config)
        else:
            return None

class GAPAttack(ICLAttackStrategy):
    def is_member_by_response(self, response: str, true_label: str) -> bool:
        # 第一种方式：判断模型是否见过输入
        duplicate_phrases = ["already", "duplicate", "same", "again", "previously", "repeat"]
        for phrase in duplicate_phrases:
            if phrase in response.lower():
                return True

        # 获取所有可能的标签
        all_labels = set(self.label_translation.values())
        
        # 如果没有翻译字典，就使用原始的输出字段值
        if not all_labels:
            all_labels = set(self.dataset['train'][self.output_field] + self.dataset['test'][self.output_field])
        
        # 检查真实标签是否在响应中，且其他标签都不在响应中
        true_label_in_response = true_label in response
        other_labels_not_in_response = all(label not in response for label in all_labels if label != true_label)
        
        return true_label_in_response and other_labels_not_in_response

    def attack(self, model: 'ModelInterface'):
        num_attacks = self.attack_config.get('num_attacks', 100)

        for i in tqdm(range(num_attacks)):
            icl_samples, icl_prompt = self.generate_icl_prompt()
            attack_sample, is_member = self.get_attack_sample(icl_samples)

            attack_prompt = icl_prompt + [{
                "role": "user",
                "content": self.get_demo_template()[0]['content'].format(input=attack_sample["input"])
            }]

            response = model.query(attack_prompt, "Question Classification")[0]
            pred_member = self.is_member_by_response(response, str(attack_sample["output"]))
            self.results.append((pred_member, is_member))

            # 添加日志输出
            logger.info(f"Attack {i+1}/{num_attacks}:")
            logger.info(f"Input: {attack_sample['input']}")
            logger.info(f"True label: {attack_sample['output']}")
            logger.info(f"Model response: {response}")
            logger.info(f"Is member: {is_member}, Predicted member: {pred_member}")
            logger.info("-" * 50)

    def evaluate(self) -> Dict[str, float]:
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        
        for pred_member, is_member in self.results:
            if pred_member and is_member:
                true_positives += 1
            elif not pred_member and not is_member:
                true_negatives += 1
            elif pred_member and not is_member:
                false_positives += 1
            else:
                false_negatives += 1
        
        total = len(self.results)
        accuracy = (true_positives + true_negatives) / total
        
        # 计算Advantage
        advantage = 2 * (accuracy - 0.5)
        
        # 计算精确率、召回率和F1分数
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "advantage": advantage,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(args):
    data_config = load_yaml_config(args.data)
    attack_config = load_yaml_config(args.attack)
    query_config = load_yaml_config(args.query)

    attack_strategy = ICLAttackStrategy.create(attack_config)
    if attack_strategy is None:
        raise ValueError(f"Attack type {attack_config['type']} is not supported.")
    model = ModelInterface(query_config)
    attack_strategy.prepare(data_config)
    try:
        attack_strategy.attack(model)
    except KeyboardInterrupt:
        pass
    results = attack_strategy.evaluate()
    print(f"Attack results: {results}")

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='Path to the data config file', default="data.yaml")
    parser.add_argument('--attack', help='Path to the attack config file', default="attack.yaml")
    parser.add_argument('--query', help='Path to the query config file', default="query.yaml")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='WARNING', help="Set the logging level")
    args = parser.parse_args()
    
    logger = get_logger("ICL Attack", args.log_level)

    main(args)