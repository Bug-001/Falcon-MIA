import argparse
import yaml
import itertools
from datetime import datetime
import importlib
import os
from typing import Dict, List, Any
import copy
import concurrent.futures
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import json
import multiprocessing
from multiprocessing import current_process
import signal

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True

@dataclass
class ExperimentTask:
    """实验任务类"""
    exp_name: str
    params: List[Dict]
    group_key: Tuple  # 用于标识实验所属组
    dependencies: Set[Tuple]  # 依赖的组

class ExperimentRunner:
    def __init__(self, params_config: Dict[str, Any]):
        """初始化实验运行器"""
        self.config = params_config
        
        # 加载主函数
        self.main_func = self._load_main_function()
        
        # 加载yaml模板
        self.templates = self._load_templates()

        # 获取程序文件的最后修改时间
        self.last_modified_time = self._get_program_last_modified_time()
        
        # 并行配置
        parallel_config = self.config.get('parallel', {})
        self.parallel_enabled = parallel_config.get('enable', False)
        self.max_workers = parallel_config.get('max_workers', 1) if self.parallel_enabled else 1

        self.killer = GracefulKiller()

    def _load_main_function(self):
        """获取程序主函数句柄"""
        module_name = self.config['program']['module']
        function_name = self.config['program']['function']
        
        try:
            module = importlib.import_module(module_name)
            return getattr(module, function_name)
        except Exception as e:
            raise ImportError(f"Failed to load main function: {str(e)}")
            
    def _load_templates(self) -> Dict[str, Dict]:
        """加载所有yaml模板"""
        templates = []
        for param_config in self.config['params']:
            template_path = param_config['param']
            with open(template_path, 'r') as f:
                template_data = yaml.safe_load(f)
                templates.append(template_data)
        return templates
        
    def _generate_experiment_name(self, param_values: Dict[str, Dict[str, Any]]) -> str:
        """生成实验名称"""
        name_parts = []
        # 添加参数值
        for params in param_values:
            for param_name, value in params.items():
                # Sanitize the value to avoid special characters
                value = str(value).replace("/", "_")
                if param_name == 'model' and value.endswith(".yaml"):
                    # XXX: If the model name is *.yaml, just ignore it
                    continue
                name_parts.append(f"{param_name}({value})")

        return self.config['name_prefix'] + "/" + "--".join(name_parts)
        
    def _is_valid_combination(self, params: Dict[str, Dict[str, Any]]) -> bool:
        """检查参数组合是否满足约束条件"""
        if 'constraints' not in self.config:
            return True
            
        # 创建本地变量以便eval执行约束条件
        locals_dict = {'params': params}
        
        for constraint in self.config['constraints']:
            try:
                if not eval(constraint, {}, locals_dict):
                    return False
            except Exception as e:
                print(f"Warning: Failed to evaluate constraint '{constraint}': {str(e)}")
                return False
                
        return True
        
    def _generate_param_combinations(self):
        """生成所有有效的参数组合"""
        all_param_combinations = []
        
        # 为每个参数配置生成组合
        for param_config in self.config['params']:
            param_specs = param_config.get('config', {})
            if not param_specs:
                all_param_combinations.append([{}])
                continue
                
            combinations = self._process_param_dict(param_specs)
            all_param_combinations.append(combinations)
        
        # 将不同文件的参数直接组合起来
        for param_combo in itertools.product(*all_param_combinations):
            if self._is_valid_combination(list(param_combo)):
                yield list(param_combo)
    
    def _process_param_dict(self, param_dict):
        """递归处理参数字典，生成所有可能的组合"""
        all_choices = []
        
        for param_name, param_values in param_dict.items():
            cur_choices = []

            # param_values is a list, in which every elem can be the value, or a value with dependent param dict
            for value_spec in param_values:
                # 如果是普通元素
                if not isinstance(value_spec, dict):
                    cur_choices.append({param_name: value_spec})

                # 如果是带依赖的配置列表
                else:
                    value = value_spec['value']
                    dep = value_spec['dependency']
                    dep_combinations = self._process_param_dict(dep)
                    for dep_combo in dep_combinations:
                        dep_combo[param_name] = value
                        cur_choices.append(dep_combo)
                
            all_choices.append(cur_choices)
        
        ret = []
        for param_dicts in itertools.product(*all_choices):
            cur_dict = dict()
            for param_dict in param_dicts:
                cur_dict = cur_dict | param_dict
            ret.append(cur_dict)
        return ret
    
    def _get_program_last_modified_time(self) -> float:
        """获取程序文件的最后修改时间"""
        module_name = self.config['program']['module']
        try:
            module = importlib.import_module(module_name)
            module_file = module.__file__
            return os.path.getmtime(module_file)
        except Exception as e:
            print(f"Warning: Failed to get program modification time: {str(e)}")
            return 0
            
    def _get_timestamp_from_name(self, exp_name: str) -> datetime:
        """从实验名称中提取时间信息"""
        try:
            # 假设时间戳格式为YYYYMMDDHHmm
            timestamp_str = ""
            for part in exp_name.split("-"):
                if len(part) == 12 and part.isdigit():  # 找到符合格式的时间戳部分
                    timestamp_str = part
                    break
            
            if timestamp_str:
                return datetime.strptime(timestamp_str, "%Y%m%d%H%M")
            else:
                # 如果没找到时间戳，返回一个很早的时间以确保重新运行
                return datetime.min
        except Exception as e:
            print(f"Warning: Failed to parse timestamp from {exp_name}: {str(e)}")
            return datetime.min

    def _need_rerun(self, exp_name: str) -> bool:
        """判断实验是否需要重新运行"""
        exp_dir = Path(os.path.join("cache/log", exp_name))

        # 检查metrics.json是否为空或损坏
        metric_file = os.path.join(exp_dir, "metrics.json")
        try:
            with open(metric_file, 'r') as f:
                metrics_data = json.load(f)
            if not metrics_data:  # 如果文件为空字典
                print(f"Running experiment {exp_name} (metrics file is empty)")
                return True
        except (json.JSONDecodeError, IOError):
            print(f"Running experiment {exp_name} (metrics file is not available)")
            return True

        print(f"Skipping experiment {exp_name} (already completed)")
        return False
        
    def _update_experiment_name(self, old_name: str) -> str:
        """更新实验名称中的时间戳"""
        parts = old_name.split("-")
        new_timestamp = datetime.now().strftime("%Y%m%d%H%M")
        
        # 更新或插入时间戳
        timestamp_updated = False
        for i, part in enumerate(parts):
            if len(part) == 12 and part.isdigit():  # 找到原时间戳
                parts[i] = new_timestamp
                timestamp_updated = True
                break
                
        if not timestamp_updated:  # 如果没找到原时间戳，在prefix后添加
            prefix_index = parts.index(self.config['name_prefix']) if self.config['name_prefix'] in parts else 0
            parts.insert(prefix_index + 1, new_timestamp)
            
        return "-".join(parts)

    def _prepare_experiment_batch(self) -> List[Tuple[List[Dict], str]]:
        """准备实验批次"""
        experiment_batch = []
        for param_combo in self._generate_param_combinations():
            experiment_batch.append(param_combo)
        return experiment_batch

    def _run_single_experiment(self, param_combo: List[Dict]):
        """运行单个实验的完整流程"""
        exp_name = self._generate_experiment_name(param_combo)

        # 检查是否需要重新运行
        if not self._need_rerun(exp_name):
            return

        # 如果需要重新运行，且目录已存在，更新实验名称
        if Path(exp_name).exists():
            new_exp_name = self._update_experiment_name(exp_name)
            print(f"[Process {current_process().name}] Updating experiment name from {exp_name} to {new_exp_name}")
            exp_name = new_exp_name
            
        print(f"\n[Process {current_process().name}] Running experiment: {exp_name}")
        
        # 准备参数
        current_params = []
        for template, params in zip(self.templates, param_combo):
            template_copy = copy.deepcopy(template)
            template_copy.update(params)
            current_params.append(template_copy)

        # 运行实验
        try:
            self.main_func(*current_params, exp_name)
            print(f"[Process {current_process().name}] Experiment {exp_name} completed successfully")
        except Exception as e:
            print(f"[Process {current_process().name}] Error in experiment {exp_name}: {str(e)}")
            raise

    def run(self):
        """运行所有实验（支持并行）"""
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for param_combo in self._generate_param_combinations():
                    if self.killer.kill_now:
                        break
                    futures.append(
                        executor.submit(self._run_single_experiment, param_combo)
                    )

                for future in concurrent.futures.as_completed(futures):
                    if self.killer.kill_now:
                        print("\nShutting down gracefully...")
                        # 取消所有未完成的任务
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        future.result(timeout=60)  # 添加超时限制
                    except concurrent.futures.TimeoutError:
                        print("A task timed out")
                    except Exception as e:
                        print(f"Task failed with: {e}")

        except KeyboardInterrupt:
            print("\nCaught KeyboardInterrupt, shutting down...")
            return
        finally:
            # 确保清理所有资源
            print("Cleaning up resources...")

def main(params):    
    runner = ExperimentRunner(params)
    runner.run()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', default='params.yaml', help='Path to parameters config file')
    args = parser.parse_args()
    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)
    main(params)