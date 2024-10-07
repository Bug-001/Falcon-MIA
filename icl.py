# icl_attack.py

import yaml
import argparse
import random
import string
import numpy as np
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from scipy.spatial.distance import cosine, euclidean, cityblock
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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

class EvaluationMetrics:
    @staticmethod
    def calculate_advantage(predictions, ground_truth):
        true_positives = sum((p == 1 and g == 1) for p, g in zip(predictions, ground_truth))
        true_negatives = sum((p == 0 and g == 0) for p, g in zip(predictions, ground_truth))
        false_positives = sum((p == 1 and g == 0) for p, g in zip(predictions, ground_truth))
        false_negatives = sum((p == 0 and g == 1) for p, g in zip(predictions, ground_truth))
        
        accuracy = (true_positives + true_negatives) / len(predictions)
        advantage = 2 * (accuracy - 0.5)
        
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
    
    @staticmethod
    def calculate_roc_auc(ground_truth, scores):
        fpr, tpr, _ = roc_curve(ground_truth, scores)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    @staticmethod
    def calculate_log_roc_auc(ground_truth, scores):
        fpr, tpr, _ = roc_curve(ground_truth, scores)
        log_fpr = np.logspace(-8, 0, num=100)
        log_tpr = np.interp(log_fpr, fpr, tpr)
        log_auc = auc(log_fpr, log_tpr)
        return log_fpr, log_tpr, log_auc

    @staticmethod
    def plot_log_roc(log_fpr, log_tpr, log_auc, filename='log_roc_curve.png'):
        plt.figure()
        plt.plot(log_fpr, log_tpr, color='darkorange', lw=2, label=f'Log ROC curve (AUC = {log_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xscale('log')
        plt.xlim([1e-8, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Logarithmic Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_roc(fpr, tpr, roc_auc, filename='roc_curve.png'):
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(filename)
        plt.close()

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
    
    def remove_punctuation(self, word):
        return word.strip(string.punctuation)

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
        elif attack_type == 'Inquiry':
            return InquiryAttack(attack_config)
        elif attack_type == 'Repeat':
            return RepeatAttack(attack_config)
        elif attack_type == 'Brainwash':
            return BrainwashAttack(attack_config)
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

            final_prompt = icl_prompt + [{
                "role": "user",
                "content": self.get_demo_template()[0]['content'].format(input=attack_sample["input"])
            }]
            response = model.query(final_prompt, "Question Classification")[0]
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
        predictions = [int(pred) for pred, _ in self.results]
        ground_truth = [int(truth) for _, truth in self.results]
        return EvaluationMetrics.calculate_advantage(predictions, ground_truth)

class InquiryAttack(ICLAttackStrategy):
    def __init__(self, attack_config):
        super().__init__(attack_config)
        self.inquiry_template = attack_config.get('inquiry_template', "Have you seen this sentence before: {sample}?")
        self.positive_keywords = attack_config.get('positive_keywords', ["yes", "seen", "encountered", "familiar"])
        self.negative_keywords = attack_config.get('negative_keywords', ["no", "not seen", "unfamiliar"])

    def construct_inquiry(self, sample):
        return self.inquiry_template.format(sample=sample)

    def is_member_by_response(self, response):
        words = [self.remove_punctuation(word.lower()) for word in response.split()]

        if len(words) == 0:
            return None

        # 检查负面关键词
        if any(word in self.negative_keywords for word in words):
            return False
        if "have not seen" in response.lower() or "haven't seen" in response.lower():
            return False
        
        # 检查正面关键词
        if any(word in self.positive_keywords for word in words):
            return True
        if "have seen" in response.lower() or "have encountered" in response.lower():
            return True
        
        if words[0].startswith("1"):
            return True
        elif words[0].startswith("0"):
            return False

        # 模型未给出有效信息
        return None

    def attack(self, model):
        self.results = []
        for i in tqdm(range(self.attack_config.get('num_attacks', 100))):
            icl_samples, icl_prompt = self.generate_icl_prompt()
            attack_sample, is_member = self.get_attack_sample(icl_samples)
            
            final_prompt = icl_prompt + [{
                "role": "user",
                "content": self.construct_inquiry(attack_sample["input"])
            }]
            response = model.query(final_prompt, "Inquiry Attack")[0]
            
            pred_member = self.is_member_by_response(response)
            if pred_member is not None:
                self.results.append((pred_member, is_member))
            else:
                # 对于无效的响应，以50%的概率随机选择
                self.results.append((random.random() < 0.5, is_member))
            
            # 添加日志输出
            logger.info(f"Attack {i+1}/{self.attack_config.get('num_attacks', 100)}:")
            logger.info(f"Sample: {attack_sample['input']}")
            logger.info(f"Model response: {response}")
            logger.info(f"Is member: {is_member}, Predicted member: {pred_member}")
            logger.info("-" * 50)

    def evaluate(self):
        predictions = [int(pred) for pred, _ in self.results]
        ground_truth = [int(truth) for _, truth in self.results]
        return EvaluationMetrics.calculate_advantage(predictions, ground_truth)

class RepeatAttack(ICLAttackStrategy):
    def __init__(self, attack_config):
        super().__init__(attack_config)
        self.num_words = attack_config.get('num_words', 3)
        self.similarity_threshold = attack_config.get('similarity_threshold', 0.8)
        self.similarity_metric = attack_config.get('similarity_metric', 'cosine')
        self.encoder = None  # 延迟初始化编码器

    def initialize_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder_name = self.attack_config.get('encoder', 'paraphrase-MiniLM-L6-v2')
            self.encoder = SentenceTransformer(encoder_name)

    def truncate_sample(self, sample, num_words):
        return ' '.join(sample.split()[:num_words])

    def calculate_similarity(self, original, generated):
        self.initialize_encoder()  # 确保编码器已初始化
        original_embedding = self.encoder.encode([original])[0]
        generated_embedding = self.encoder.encode([generated])[0]
        
        if self.similarity_metric == 'cosine':
            return 1 - cosine(original_embedding, generated_embedding)
        elif self.similarity_metric == 'euclidean':
            return 1 / (1 + euclidean(original_embedding, generated_embedding))
        elif self.similarity_metric == 'manhattan':
            return 1 / (1 + cityblock(original_embedding, generated_embedding))
        elif self.similarity_metric == 'dot_product':
            return np.dot(original_embedding, generated_embedding) / (np.linalg.norm(original_embedding) * np.linalg.norm(generated_embedding))
        elif self.similarity_metric == 'jaccard':
            # 对于Jaccard相似度，我们需要将向量转换为集合
            set1 = set(np.where(original_embedding > np.mean(original_embedding))[0])
            set2 = set(np.where(generated_embedding > np.mean(generated_embedding))[0])
            return len(set1.intersection(set2)) / len(set1.union(set2))
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

    def attack(self, model):
        self.results = []
        for i in tqdm(range(self.attack_config.get('num_attacks', 100))):
            icl_samples, icl_prompt = self.generate_icl_prompt()
            attack_sample, is_member = self.get_attack_sample(icl_samples)
            
            # Take num_words as the half length of the sentence if num_words is 0
            num_words = self.num_words if self.num_words > 0 else len(attack_sample["input"]) // 2
            truncated_input = self.truncate_sample(attack_sample["input"], num_words)
            all_prompt = icl_prompt + [{
                "role": "user",
                "content": self.attack_config.get('repeat_template', "Complete the following sentence: {sample}").format(sample=truncated_input)
            }]
            generated_text = model.query(all_prompt, "Repeat Attack")[0]
            
            similarity_1 = self.calculate_similarity(attack_sample["input"], generated_text)
            similarity_2 = self.calculate_similarity(attack_sample["input"][num_words:], generated_text)
            similarity = max(similarity_1, similarity_2)
            pred_member = similarity >= self.similarity_threshold
            
            self.results.append((pred_member, is_member, similarity))
            
            # 添加日志输出
            logger.info(f"Attack {i+1}/{self.attack_config.get('num_attacks', 100)}:")
            logger.info(f"Original: {attack_sample['input']}")
            logger.info(f"Generated: {generated_text}")
            logger.info(f"Similarity: {similarity}")
            logger.info(f"Is member: {is_member}, Predicted member: {pred_member}")
            logger.info("-" * 50)

    def evaluate(self):
        predictions = [int(pred) for pred, _, _ in self.results]
        ground_truth = [int(truth) for _, truth, _ in self.results]
        similarities = [sim for _, _, sim in self.results]
        
        metrics = EvaluationMetrics.calculate_advantage(predictions, ground_truth)
        metrics['average_similarity'] = np.mean(similarities)
        
        # 计算ROC曲线和AUC
        fpr, tpr, roc_auc = EvaluationMetrics.calculate_roc_auc(ground_truth, similarities)
        metrics['auc'] = roc_auc

        # 计算log ROC
        log_fpr, log_tpr, log_auc = EvaluationMetrics.calculate_log_roc_auc(ground_truth, similarities)
        metrics['log_auc'] = log_auc

        # 存储ROC和log ROC数据
        self.roc_data = {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'log_fpr': log_fpr,
            'log_tpr': log_tpr,
            'log_auc': log_auc
        }

        # 绘制ROC和log ROC曲线
        if self.attack_config.get('plot_roc', False):
            EvaluationMetrics.plot_roc(fpr, tpr, roc_auc, f'roc_curve_{self.__class__.__name__}.png')
        if self.attack_config.get('plot_log_roc', False):
            EvaluationMetrics.plot_log_roc(log_fpr, log_tpr, log_auc, f'log_roc_curve_{self.__class__.__name__}.png')

        return metrics

class BrainwashAttack(ICLAttackStrategy):
    def __init__(self, attack_config: Dict[str, Any]):
        super().__init__(attack_config)
        self.max_misleadings = attack_config.get('max_misleadings', 10)
        self.threshold = attack_config.get('brainwash_threshold', 5)
        self.num_wrong_labels = attack_config.get('num_wrong_labels', 3)  # 新增参数

    def brainwashed(self, response: str, wrong_label: str) -> bool:
        # 获取所有可能的标签
        other_labels = {label.lower() for label in self.label_translation.values()}
        wrong_label = wrong_label.lower()
        other_labels.remove(wrong_label)

        # 断句
        words = [self.remove_punctuation(word.lower()) for word in response.split()]
        
        # 检查错误标签是否在响应中，且其他标签都不在响应中
        wrong_label_in_response = any(wrong_label in word for word in words)
        other_labels_not_in_response = all(all(label not in word for word in words) for label in other_labels)
        
        return wrong_label_in_response and other_labels_not_in_response

    def binary_search_iterations(self, model: ModelInterface, prompt: List[Dict[str, str]], 
                                 attack_sample: Dict[str, str], wrong_label: str) -> int:
        def generate_prompt_and_request(iterations):
            query_prompt = prompt.copy()
            for _ in range(iterations):
                query_prompt = query_prompt + [{
                    "role": "user",
                    "content": self.user_prompt.format(input=attack_sample["input"])
                }, {
                    "role": "assistant",
                    "content": self.assistant_prompt.format(output=wrong_label)
                }]
            query_prompt = query_prompt + [{
                "role": "user",
                "content": self.user_prompt.format(input=attack_sample["input"])
            }]
            return model.query(query_prompt, "Brainwash Attack")[0]
        
        left, right = 0, self.max_misleadings
        mid = self.max_misleadings
        while left < right:
            old_mid = mid
            response = generate_prompt_and_request(mid)
            if self.brainwashed(response, wrong_label):
                right = mid
            else:
                left = mid + 1
            mid = (left + right) // 2
        final_response = generate_prompt_and_request(old_mid)
        if old_mid != self.max_misleadings:
            while not self.brainwashed(final_response, wrong_label):
                old_mid += 1
                final_response = generate_prompt_and_request(old_mid)
            logger.info(f"Brainwashed to \"{wrong_label}\" in {old_mid} turns: {final_response}")
        else:
            logger.info(f"Failed to brainwash to \"{wrong_label}\" in {old_mid} turns: {final_response}")
        return old_mid

    def attack(self, model: ModelInterface):
        self.results = []
        self.scores = []

        demo_template = self.get_demo_template()
        self.user_prompt = demo_template[0]['content']
        self.assistant_prompt = demo_template[1]['content']
        
        for _ in tqdm(range(self.attack_config['num_attacks'])):
            icl_samples, icl_prompt = self.generate_icl_prompt()
            attack_sample, is_member = self.get_attack_sample(icl_samples)
            
            correct_label = attack_sample["output"]
            wrong_labels = [label for label in self.label_translation.values() if label != correct_label]
            selected_wrong_labels = random.sample(wrong_labels, min(self.num_wrong_labels, len(wrong_labels)))

            logger.info(f"Attack {_+1}/{self.attack_config['num_attacks']}:")
            logger.info(f"Sample: {attack_sample['input']}")
            logger.info(f"Correct label: {correct_label}")

            prompt = icl_prompt.copy()
            iterations = []
            for wrong_label in selected_wrong_labels:
                iteration = self.binary_search_iterations(model, icl_prompt, attack_sample, wrong_label)
                iterations.append(iteration)
            
            avg_iterations = np.mean(iterations)
            pred_member = avg_iterations >= self.threshold
            self.results.append((pred_member, is_member))
            self.scores.append(avg_iterations)
            
            # 添加日志输出
            logger.info(f"Iterations: {iterations}")
            logger.info(f"Is member: {is_member}, Predicted member: {pred_member}")
            logger.info("-" * 50)

    def evaluate(self) -> Dict[str, float]:
        predictions = [int(pred) for pred, _ in self.results]
        ground_truth = [int(truth) for _, truth in self.results]
        
        metrics = EvaluationMetrics.calculate_advantage(predictions, ground_truth)
        
        fpr, tpr, roc_auc = EvaluationMetrics.calculate_roc_auc(ground_truth, self.scores)
        log_fpr, log_tpr, log_auc = EvaluationMetrics.calculate_log_roc_auc(ground_truth, self.scores)
        
        metrics['auc'] = roc_auc
        metrics['log_auc'] = log_auc

        if self.attack_config.get('plot_roc', False):
            EvaluationMetrics.plot_roc(fpr, tpr, roc_auc, 'brainwash_roc_curve.png')
            EvaluationMetrics.plot_log_roc(log_fpr, log_tpr, log_auc, 'brainwash_log_roc_curve.png')

        return metrics

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