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
from datetime import datetime
from colorama import Fore, Style
from datasets import Dataset, DatasetDict

from data import DATASET_LOADERS, BaseDataLoader

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
        self.output_dir = os.path.join(output_dir, "log", name)
        os.makedirs(self.output_dir, exist_ok=True)
        
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

    def load(self, filename: str) -> None:
        """从文件加载表格"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")
            
        df = pd.read_json(filename, orient='records')
        table_name = filename  # 从文件名提取表名
        self._tables[table_name] = df
        self._row_counters[table_name] = len(df)
        self._current_table = table_name

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
        
        self._dataset, self.config = loader.load(task)

    def get_config(self) -> Dict[str, Any]:
        return self.config

    def get_available_tasks(self) -> List[str]:
        loader: BaseDataLoader = DATASET_LOADERS[self.dataset_name]()
        return loader.get_supported_tasks()

    def crop_dataset(self, num=-1, split=[0.8,0.1,0.1], seed=42):
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

        dataset: DatasetDict = self._dataset.copy()

        train_data = dataset['train'].shuffle(seed=seed)
        train_dataset = train_data.select(range(train_num))

        if 'validation' in dataset.keys():
            val_data = dataset['validation'].shuffle(seed=seed)
            val_dataset = val_data.select(range(val_num))
        else:
            val_dataset = train_data.select(range(train_num, train_num + val_num))

        if 'test' in dataset.keys():
            test_data = dataset['test'].shuffle(seed=seed)
            test_dataset = test_data.select(range(test_num))
        else:
            test_dataset = train_data.select(range(train_num + val_num, train_num + val_num + test_num))
        
        # 创建DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        return dataset_dict
    
    def save_dataset(self, path: str, dataset: DatasetDict = None) -> None:
        """保存数据集到文件"""
        if dataset == None:
            dataset = self._dataset
        dataset_path = os.path.join(self.dir, path)
        dataset.save_to_disk(dataset_path)

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