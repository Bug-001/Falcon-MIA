import os
import numpy as np
import json
import pickle
import functools
import hashlib
from typing import Any, Callable
from matplotlib import pyplot as plt
from contextlib import contextmanager
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
from colorama import Fore, Style

# Global output directory
_cwd_stack = []

@contextmanager
def output_directory(path: str = None):
    """Temporarily change the output directory
    
    Example:
        with output_directory('new/path'):
            # Do something with new output path
        # Output path is restored
    """
    global _cwd_stack

    if path is None:
        path = "unnamed_experiment"
    
    # Convert to absolute path and ensure it exists
    temp_dir = os.path.abspath(os.path.join("output", path))
    os.makedirs(temp_dir, exist_ok=True)
    _cwd_stack.append(os.getcwd())
    os.chdir(temp_dir)
    
    try:
        yield
    finally:
        if len(_cwd_stack) > 0:
            os.chdir(_cwd_stack[-1])
            _cwd_stack.pop()

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

class ExperimentLogger:
    def __init__(self):
        self._tables: Dict[str, pd.DataFrame] = {}  # 存储所有表格
        self._current_table: Optional[str] = None   # 当前选中的表格
        self._current_row: Optional[int] = None     # 当前选中的行
        self._row_counters: Dict[str, int] = {}    # 每个表格的行计数
        
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
            filename = f"{table_name}.json"
            df.to_json(filename, orient='records', indent=4)

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

def save_json(filename: str, data: Any):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)