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
        exp_dir = Path(os.path.join("output", exp_name))

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
        
    def run(self):
        """运行所有实验"""
        for param_combo in self._generate_param_combinations():
            # 生成实验名称
            exp_name = self._generate_experiment_name(param_combo)
            
            # 检查是否需要重新运行
            if not self._need_rerun(exp_name):
                continue
                
            # 如果需要重新运行，且目录已存在，更新实验名称
            if Path(exp_name).exists():
                new_exp_name = self._update_experiment_name(exp_name)
                print(f"Updating experiment name from {exp_name} to {new_exp_name}")
                exp_name = new_exp_name
                
            print(f"\nRunning experiment: {exp_name}")
            
            # 准备每个模板的参数
            current_params = copy.deepcopy(self.templates)

            # 准备参数
            current_params = []
            for template, params in zip(self.templates, param_combo):
                template_copy = copy.deepcopy(template)
                template_copy.update(params)
                current_params.append(template_copy)

            # 运行实验
            self.main_func(*current_params, exp_name)
            print(f"Experiment {exp_name} completed successfully")

class ParallelExperimentRunner(ExperimentRunner):
    def __init__(self, params_file: str = 'params.yaml'):
        super().__init__(params_file)
        self.parallel_config = self.config.get('parallel', {})
        
    def _get_group_key(self, params: Dict) -> Tuple:
        """根据分组策略生成组标识"""
        if not self.parallel_config.get('groups'):
            return tuple()
            
        group_keys = []
        for group_fields in self.parallel_config['groups']:
            key_parts = []
            for field in group_fields:
                # 支持嵌套字段访问，如 "attack_chat.type"
                value = params
                for part in field.split('.'):
                    value = value.get(part, '')
                key_parts.append(str(value))
            group_keys.append(tuple(key_parts))
            
        return tuple(group_keys)
        
    def _get_dependencies(self, group_key: Tuple) -> Set[Tuple]:
        """获取组间依赖"""
        deps = set()
        if not self.parallel_config.get('dependencies'):
            return deps
            
        for dep in self.parallel_config['dependencies']:
            if str(group_key) == dep['before']:
                deps.add(dep['after'])
        return deps
        
    def _prepare_tasks(self) -> List[ExperimentTask]:
        """准备实验任务"""
        tasks = []
        for param_combo in self._generate_param_combinations():
            exp_name = self._generate_experiment_name(param_combo)
            group_key = self._get_group_key(param_combo)
            dependencies = self._get_dependencies(group_key)
            
            tasks.append(ExperimentTask(
                exp_name=exp_name,
                params=param_combo,
                group_key=group_key,
                dependencies=dependencies
            ))
        return tasks
        
    def _run_task(self, task: ExperimentTask):
        """运行单个实验任务"""
        exp_name = task.exp_name
        
        # 检查是否需要重新运行
        if not self._need_rerun(exp_name):
            return True
            
        # 如果需要重新运行，且目录已存在，更新实验名称
        if Path(exp_name).exists():
            new_exp_name = self._update_experiment_name(exp_name)
            print(f"Updating experiment name from {exp_name} to {new_exp_name}")
            exp_name = new_exp_name
            
        print(f"\nRunning experiment: {exp_name}")
        
        template_params = []
        for template_name, params in task.params.items():
            current_params = copy.deepcopy(self.templates[template_name])
            for param_name, value in params.items():
                current_params[param_name] = value
            template_params.append(current_params)
            
        try:
            elf.main_func(*template_params, exp_name)
            print(f"Experiment {exp_name} completed successfully")
            return True
        except Exception as e:
            print(f"Experiment {exp_name} failed: {str(e)}")
            return False
            
    def run(self):
        """运行所有实验"""
        if not self.parallel_config.get('enable', False):
            # 如果未启用并行，使用原来的串行方式
            super().run()
            return
            
        # 准备任务
        tasks = self._prepare_tasks()
        
        # 按组分类任务
        tasks_by_group = defaultdict(list)
        for task in tasks:
            tasks_by_group[task.group_key].append(task)
            
        # 创建已完成组的集合
        completed_groups = set()
        
        # 获取最大并行数
        max_workers = self.parallel_config.get('max_workers', 4)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            while tasks_by_group:
                # 找出可以执行的组（所有依赖都已完成）
                ready_groups = []
                for group_key in tasks_by_group.keys():
                    tasks = tasks_by_group[group_key]
                    if all(dep in completed_groups for task in tasks for dep in task.dependencies):
                        ready_groups.append(group_key)
                
                if not ready_groups:
                    raise RuntimeError("Circular dependency detected or no tasks can proceed")
                
                # 并行执行每个就绪组内的任务
                futures = []
                for group_key in ready_groups:
                    group_tasks = tasks_by_group[group_key]
                    for task in group_tasks:
                        future = executor.submit(self._run_task, task)
                        futures.append(future)
                
                # 等待当前批次完成
                concurrent.futures.wait(futures)
                
                # 更新完成状态
                for group_key in ready_groups:
                    completed_groups.add(group_key)
                    del tasks_by_group[group_key]

def main(params):    
    runner = ParallelExperimentRunner(params)
    runner.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', default='params.yaml', help='Path to parameters config file')
    args = parser.parse_args()
    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)
    main(params)