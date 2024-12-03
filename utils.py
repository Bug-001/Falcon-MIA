import os
import numpy as np
import json
import pickle
import functools
import hashlib
from typing import Any, Callable, Tuple
from matplotlib import pyplot as plt
from contextlib import contextmanager
from typing import Dict, Any, Optional, List
import pandas as pd
import torch
from datetime import datetime
from colorama import Fore, Style, init
from datasets import Dataset, DatasetDict, concatenate_datasets

from data import DATASET_LOADERS, BaseDataLoader

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc, f1_score

# Global output directory
output_dir = "cache"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class SLogger:
    def __init__(self, name):
        self._tables: Dict[str, pd.DataFrame] = {}  # 存储所有表格
        self._current_table: Optional[str] = None   # 当前选中的表格
        self._current_row: Optional[int] = None     # 当前选中的行
        self._row_counters: Dict[str, int] = {}    # 每个表格的行计数
        self.root_dir = output_dir
        self.output_dir = os.path.join(output_dir, "log", name)
        os.makedirs(self.output_dir, exist_ok=True)
        init()
        
    def new_table(self, table_name: str) -> None:
        """创建新表格并将其设为当前表格"""
        if table_name in self._tables:
            raise ValueError(f"Table {table_name} already exists")
        self._tables[table_name] = pd.DataFrame()
        self._row_counters[table_name] = 0
        self._current_table = table_name
        self._current_row = None

    def new_row(self, table_name: Optional[str] = None) -> int:
        """在指定表格中创建新行，返回行号"""
        table = table_name or self._current_table
        if table is None:
            raise ValueError("No table specified or selected")
        if table not in self._tables:
            raise ValueError(f"Table {table} does not exist")
            
        row_idx = self._row_counters[table]
        self._row_counters[table] += 1
        self._current_table = table
        self._current_row = row_idx
        return row_idx

    def select_table(self, table_name: str) -> None:
        """选择当前表格"""
        if table_name not in self._tables:
            raise ValueError(f"Table {table_name} does not exist")
        self._current_table = table_name
        self._current_row = None

    def add(self, key: str, value: Any, 
                 table: Optional[str] = None, 
                 row: Optional[int] = None, show: Optional[bool] = True) -> None:
        if table is None and row is None:
            table = self._current_table
            row = self._current_row
        elif table is not None and row is None:
            row = self._row_counters[table] - 1
        elif table is not None and row is not None:
            pass
        else:
            raise ValueError("Table must be specified if row is specified")

        if table is not None and row is not None:
            if key not in self._tables[table].columns:
                self._tables[table][key] = None
            self._tables[table].at[row, key] = value

        if show:
            color = Fore.GREEN if isinstance(value, bool) and value else Fore.RED
            print(f"{key}: {color}{value}{Style.RESET_ALL}")

    def get_value(self, key: str, table: Optional[str] = None, 
                 row: Optional[int] = None) -> Any:
        """获取指定表格指定行的值"""
        table_name = table or self._current_table
        if table_name is None:
            raise ValueError("No table specified or selected")
            
        if row is not None:
            return self._tables[table_name].at[row, key]
        return self._tables[table_name][key]

    def get_table(self, table_name: Optional[str] = None) -> pd.DataFrame:
        """获取指定表格"""
        table = table_name or self._current_table
        if table is None:
            raise ValueError("No table specified or selected")
        try:
            # 如果表格不存在，试着用load打开
            if table not in self._tables:
                self.load(table)
        except FileNotFoundError:
        # 表格不存在，创建空表
            self.new_table(table)
        return self._tables[table]

    def get_row(self, row: int, table: Optional[str] = None) -> pd.Series:
        """获取指定表格的指定行"""
        table_name = table or self._current_table
        if table_name is None:
            raise ValueError("No table specified or selected")
        return self._tables[table_name].iloc[row]

    def save(self) -> None:
        """保存所有表格到文件"""
        for table_name, df in self._tables.items():
            filename = os.path.join(self.output_dir, f"{table_name}.json")
            df.to_json(filename, orient='records', indent=4)

    def savefig(self, *args, **kwargs) -> None:
        """保存图表到文件"""
        new_args = list(args)
        new_args[0] = os.path.join(self.output_dir, args[0])
        plt.savefig(*new_args, **kwargs)

    def save_data(self, data: Any, filename: str) -> None:
        """保存数据到文件"""
        filename = os.path.join(self.output_dir, filename)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def save_json(self, filename: str, data: Any) -> None:
        """保存数据到JSON文件"""
        filename = os.path.join(self.output_dir, filename)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)

    def save_model(self, model, filename: str) -> None:
        """保存PyTorch模型到文件"""
        filename = os.path.join(self.output_dir, filename)
        torch.save(model.state_dict(), filename)

    def load_data(self, filename: str) -> Any:
        """从文件加载数据"""
        filename = os.path.join(self.output_dir, filename)
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError:
            return None
        
    def load_model(self, model, filename: str) -> None:
        """从文件加载PyTorch模型"""
        filename = os.path.join(self.output_dir, filename)
        model.load_state_dict(torch.load(filename))

    def load(self, filename: str) -> None:
        """从文件加载表格"""
        data_path = os.path.join(self.output_dir, f"{filename}.json")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {filename} not found")
            
        df = pd.read_json(data_path, orient='records')
        table_name = filename  # 从文件名提取表名
        self._tables[table_name] = df
        self._row_counters[table_name] = len(df)
        self._current_table = table_name

    def info(self, *args, **kwargs):
        print(Fore.GREEN + "[INFO]" + Style.RESET_ALL, *args, **kwargs)

    def warning(self, *args, **kwargs):
        print(Fore.YELLOW + "[WARNING]" + Style.RESET_ALL, *args, **kwargs)

    def error(self, *args, **kwargs):
        print(Fore.RED + "[ERROR]" + Style.RESET_ALL, *args, **kwargs)

    def __str__(self) -> str:
        """返回所有表格的字符串表示"""
        result = []
        for table_name, df in self._tables.items():
            result.append(f"Table: {table_name}")
            result.append(str(df))
            result.append("-" * 50)
        return "\n".join(result)

class SDatasetManager:
    def __init__(self, dataset_name: str, task: str = "default"):
        self.dir = os.path.join(output_dir, "data")
        self.dataset_name = dataset_name

        if dataset_name not in DATASET_LOADERS:
            raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {list(DATASET_LOADERS.keys())}")
        
        loader: BaseDataLoader = DATASET_LOADERS[dataset_name]()
        supported_tasks = loader.get_supported_tasks()
        
        if task not in supported_tasks:
            raise ValueError(f"Task {task} not supported for dataset {dataset_name}. Available tasks: {supported_tasks}")
        
        self._dataset, self.config = loader.load(task, data_dir=self.dir)

    def get_config(self) -> Dict[str, Any]:
        return self.config

    def get_available_tasks(self) -> List[str]:
        loader: BaseDataLoader = DATASET_LOADERS[self.dataset_name]()
        return loader.get_supported_tasks()

    def _repeat_dataset_to_size(self, dataset, required_size):
        """将数据集重复到略大于所需大小，然后截取"""
        current_size = len(dataset)
        # 计算需要重复的次数，向上取整
        repeat_times = (required_size + current_size - 1) // current_size
        # 重复数据集
        repeated_data = concatenate_datasets([dataset] * repeat_times)
        # 截取所需大小
        ret_data = repeated_data.select(range(required_size))
        assert len(ret_data) == required_size
        return ret_data

    def get_total_size(self):
        return sum(self._dataset.num_rows.values())

    def crop_dataset(self, num=-1, split=[0.8,0.1,0.1], seed=42, 
                    prioritized_splits=['train', 'validation', 'test'], strict=False):
        # 1. 参数验证
        valid_splits = {'train', 'validation', 'test'}
        if not all(split in valid_splits for split in prioritized_splits):
            raise ValueError("prioritized_splits can only contain 'train', 'validation', or 'test'")
        
        # 2. 计算初始需求量
        if isinstance(split[0], float):
            if sum(split) != 1:
                raise ValueError("Split ratios must sum to 1")
            elif num == -1:
                num = len(self._dataset['train'])
            train_num = int(num * split[0])
            val_num = int(num * split[1])
            test_num = num - train_num - val_num
        elif isinstance(split[0], int):
            train_num, val_num, test_num = split
        
        initial_sizes = {
            'train': train_num,
            'validation': val_num,
            'test': test_num
        }
        
        # 3. 准备数据源
        if strict:
            # strict模式：每个split只能从对应名称的数据集获取
            available_data = {k: v for k, v in self._dataset.items()}
        else:
            # non-strict模式：合并所有数据
            all_data = []
            for split_name in ['train', 'validation', 'test']:
                if split_name in self._dataset:
                    all_data.append(self._dataset[split_name])
            combined_data = concatenate_datasets(all_data).shuffle(seed=seed)
            available_data = {'combined': combined_data}
        
        # 4. 计算最终分配量
        if strict:
            # strict模式：直接检查每个优先分割的数据是否足够
            for split_name in prioritized_splits:
                if split_name not in available_data:
                    raise ValueError(f"Split '{split_name}' required but not found in dataset")
                if len(available_data[split_name]) < initial_sizes[split_name]:
                    raise ValueError(
                        f"Insufficient data for prioritized split '{split_name}'. "
                        f"Required: {initial_sizes[split_name]}, "
                        f"Available: {len(available_data[split_name])}"
                    )
            final_sizes = initial_sizes
        else:
            # non-strict模式：计算实际分配量
            total_available = len(available_data['combined'])
            
            # 首先分配优先splits
            final_sizes = {}
            remaining_data = total_available
            for split_name in prioritized_splits:
                required = initial_sizes[split_name]
                if remaining_data < required:
                    raise ValueError(
                        f"Insufficient data for prioritized split '{split_name}'. "
                        f"Required: {required}, Available: {remaining_data}"
                    )
                final_sizes[split_name] = required
                remaining_data -= required
            
            # 计算非优先splits的总需求量和比例
            non_priority_splits = [s for s in valid_splits if s not in prioritized_splits]
            non_priority_total = sum(initial_sizes[s] for s in non_priority_splits)
            
            # 按比例分配剩余数据给非优先splits
            if non_priority_total > 0:  # 避免除以0
                if remaining_data < non_priority_total:
                    for split_name in non_priority_splits:
                        ratio = initial_sizes[split_name] / non_priority_total
                        final_sizes[split_name] = int(remaining_data * ratio)
                        # 处理最后一个split的舍入误差
                        if split_name == non_priority_splits[-1]:
                            final_sizes[split_name] = remaining_data - sum(
                                final_sizes.get(s, 0) for s in non_priority_splits[:-1]
                            )
                else:
                    for split_name in non_priority_splits:
                        final_sizes[split_name] = initial_sizes[split_name]
        
        # 5. 分配数据
        result_splits = {}
        if not strict:
            current_idx = 0
        
        for split_name in ['train', 'validation', 'test']:
            supplied_size = final_sizes[split_name]
            required_size = initial_sizes[split_name]
            
            if split_name in prioritized_splits:
                assert required_size == supplied_size
                if strict:
                    # strict模式：从对应split中获取数据
                    data = available_data[split_name].shuffle(seed=seed)
                    result_splits[split_name] = data.select(range(required_size))
                else:
                    # non-strict模式：从combined数据中顺序获取
                    result_splits[split_name] = available_data['combined'].select(
                        range(current_idx, current_idx + required_size)
                    )
                    current_idx += required_size
            else:
                # 非优先分割：使用数据重复策略
                if strict and split_name in available_data:
                    source_data = available_data[split_name].shuffle(seed=seed)
                else:
                    source_data = (available_data['combined']
                                .select(range(current_idx, current_idx + supplied_size)))
                    current_idx += supplied_size
                
                if supplied_size < required_size:
                    result_splits[split_name] = self._repeat_dataset_to_size(
                        source_data, required_size
                    )
                else:
                    result_splits[split_name] = source_data.select(range(required_size))
        
        return DatasetDict(result_splits)
    
    def save_dataset(self, path: str, dataset: DatasetDict = None) -> None:
        """保存数据集到文件"""
        if dataset == None:
            dataset = self._dataset
        dataset_path = os.path.join(self.dir, path)
        dataset.save_to_disk(dataset_path)

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
    def get_best_threshold(ground_truth, scores):
        # 计算ROC
        fpr, tpr, thresholds = roc_curve(ground_truth, scores)

        # 找到最佳F1分数对应的阈值
        f1_scores = [f1_score(ground_truth, scores >= threshold) for threshold in thresholds]
        best_threshold_index = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_index]
        best_f1 = f1_scores[best_threshold_index]

        return best_threshold, best_f1

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

# def save_json(filename: str, data: Any):
#     with open(filename, 'w') as f:
#         json.dump(data, f, indent=2, cls=NumpyEncoder)

# 使用示例
# if __name__ == "__main__":
#     sdm = SDatasetManager()
#     # 测试多任务数据集
#     print("SQuAD supported tasks:", sdm.get_available_tasks("squad"))
#     for task in sdm.get_available_tasks("squad"):
#         dataset, config = sdm.load_dataset_and_config("squad", task)
#         print(f"\nSQuAD {task} task config:", config)
#         print(f"SQuAD {task} example:", dataset['train'][0])
    
#     # 测试单任务数据集
#     single_task_datasets = ["gpqa", "trec", "agnews"]
#     for dataset_name in single_task_datasets:
#         print(f"\n{dataset_name} supported tasks:", sdm.get_available_tasks(dataset_name))
#         dataset, config = sdm.load_dataset_and_config(dataset_name)
#         print(f"{dataset_name} Config:", config)
#         print(f"{dataset_name} Example:", dataset['train'][0])