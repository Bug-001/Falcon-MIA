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
import traceback
class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True

@dataclass
class ExperimentTask:
    """Experiment task class"""
    exp_name: str
    params: List[Dict]
    group_key: Tuple  # Used to identify the group to which the experiment belongs
    dependencies: Set[Tuple]  # Dependent groups

class ExperimentRunner:
    def __init__(self, params_config: Dict[str, Any]):
        """Initialize experiment runner"""
        self.config = params_config
        
        # Load main function
        self.main_func = self._load_main_function()
        
        # Load yaml templates
        self.templates = self._load_templates()

        # Get the last modified time of the program file
        self.last_modified_time = self._get_program_last_modified_time()
        
        # Parallel configuration
        parallel_config = self.config.get('parallel', {})
        self.parallel_enabled = parallel_config.get('enable', False)
        self.max_workers = parallel_config.get('max_workers', 1) if self.parallel_enabled else 1

        self.killer = GracefulKiller()

    def _load_main_function(self):
        """Get main function handle"""
        module_name = self.config['program']['module']
        function_name = self.config['program']['function']
        
        try:
            module = importlib.import_module(module_name)
            return getattr(module, function_name)
        except Exception as e:
            raise ImportError(f"Failed to load main function: {str(e)}")
            
    def _load_templates(self) -> Dict[str, Dict]:
        """Load all yaml templates"""
        templates = []
        for param_config in self.config['params']:
            template_path = param_config['param']
            with open(template_path, 'r') as f:
                template_data = yaml.safe_load(f)
                templates.append(template_data)
        return templates
        
    def _generate_experiment_name(self, param_values: Dict[str, Dict[str, Any]]) -> str:
        """Generate experiment name"""
        name_parts = []
        # Add parameter values
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
        """Check if parameter combination meets constraints"""
        if 'constraints' not in self.config:
            return True
            
        # Create local variables for eval to execute constraints
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
        """Generate all valid parameter combinations"""
        all_param_combinations = []
        
        # Generate combinations for each parameter configuration
        for param_config in self.config['params']:
            param_specs = param_config.get('config', {})
            if not param_specs:
                all_param_combinations.append([{}])
                continue
                
            combinations = self._process_param_dict(param_specs)
            all_param_combinations.append(combinations)
        
        # Combine parameters directly from different files
        for param_combo in itertools.product(*all_param_combinations):
            if self._is_valid_combination(list(param_combo)):
                yield list(param_combo)
    
    def _process_param_dict(self, param_dict):
        """Recursively process parameter dictionary, generating all possible combinations"""
        all_choices = []
        
        for param_name, param_values in param_dict.items():
            cur_choices = []

            # param_values is a list, in which every elem can be the value, or a value with dependent param dict
            for value_spec in param_values:
                # If it's a normal element
                if not isinstance(value_spec, dict):
                    # Try using eval to parse parameter value
                    parsed_value = self._try_eval_value(value_spec)
                    cur_choices.append({param_name: parsed_value})

                # If it's a configuration list with dependencies
                else:
                    value = value_spec['value']
                    # Also try using eval to parse value
                    parsed_value = self._try_eval_value(value)
                    dep = value_spec['dependency']
                    dep_combinations = self._process_param_dict(dep)
                    for dep_combo in dep_combinations:
                        dep_combo[param_name] = parsed_value
                        cur_choices.append(dep_combo)
                
            all_choices.append(cur_choices)
        
        ret = []
        for param_dicts in itertools.product(*all_choices):
            cur_dict = dict()
            for param_dict in param_dicts:
                cur_dict = cur_dict | param_dict
            ret.append(cur_dict)
        return ret
    
    def _try_eval_value(self, value):
        """Try to evaluate value, return original value if failed"""
        # Only try eval on string type
        if not isinstance(value, str):
            return value
        
        try:
            # Try eval to parse string
            parsed_value = eval(value)
            return parsed_value
        except Exception:
            # If parsing fails, return original string
            return value
    
    def _get_program_last_modified_time(self) -> float:
        """Get the last modified time of the program file"""
        module_name = self.config['program']['module']
        try:
            module = importlib.import_module(module_name)
            module_file = module.__file__
            return os.path.getmtime(module_file)
        except Exception as e:
            print(f"Warning: Failed to get program modification time: {str(e)}")
            return 0
            
    def _get_timestamp_from_name(self, exp_name: str) -> datetime:
        """Extract time information from experiment name"""
        try:
            # Assume timestamp format is YYYYMMDDHHmm
            timestamp_str = ""
            for part in exp_name.split("-"):
                if len(part) == 12 and part.isdigit():  # Find timestamp part that matches format
                    timestamp_str = part
                    break
            
            if timestamp_str:
                return datetime.strptime(timestamp_str, "%Y%m%d%H%M")
            else:
                # If no timestamp found, return a very early time to ensure rerun
                return datetime.min
        except Exception as e:
            print(f"Warning: Failed to parse timestamp from {exp_name}: {str(e)}")
            return datetime.min

    def _need_rerun(self, exp_name: str) -> bool:
        """Determine if experiment needs to be rerun"""
        exp_dir = Path(os.path.join("cache/log", exp_name))

        # Check if metrics.json is empty or corrupted
        metric_file = os.path.join(exp_dir, "metrics.json")
        try:
            with open(metric_file, 'r') as f:
                metrics_data = json.load(f)
            if not metrics_data:  # If file is empty dictionary
                print(f"Running experiment {exp_name} (metrics file is empty)")
                return True
        except (json.JSONDecodeError, IOError):
            print(f"Running experiment {exp_name} (metrics file is not available)")
            return True

        print(f"Skipping experiment {exp_name} (already completed)")
        return False
        
    def _update_experiment_name(self, old_name: str) -> str:
        """Update timestamp in experiment name"""
        parts = old_name.split("-")
        new_timestamp = datetime.now().strftime("%Y%m%d%H%M")
        
        # Update or insert timestamp
        timestamp_updated = False
        for i, part in enumerate(parts):
            if len(part) == 12 and part.isdigit():  # Find original timestamp
                parts[i] = new_timestamp
                timestamp_updated = True
                break
                
        if not timestamp_updated:  # If original timestamp not found, add to prefix
            prefix_index = parts.index(self.config['name_prefix']) if self.config['name_prefix'] in parts else 0
            parts.insert(prefix_index + 1, new_timestamp)
            
        return "-".join(parts)

    def _prepare_experiment_batch(self) -> List[Tuple[List[Dict], str]]:
        """Prepare experiment batch"""
        experiment_batch = []
        for param_combo in self._generate_param_combinations():
            experiment_batch.append(param_combo)
        return experiment_batch

    def _run_single_experiment(self, param_combo: List[Dict]):
        """Run complete process for single experiment"""
        exp_name = self._generate_experiment_name(param_combo)

        # Check if needs to be rerun
        if not self._need_rerun(exp_name):
            return

        # If needs to be rerun and directory exists, update experiment name
        if Path(exp_name).exists():
            new_exp_name = self._update_experiment_name(exp_name)
            print(f"[Process {current_process().name}] Updating experiment name from {exp_name} to {new_exp_name}")
            exp_name = new_exp_name
            
        print(f"\n[Process {current_process().name}] Running experiment: {exp_name}")
        
        # Prepare parameters
        current_params = []
        for template, params in zip(self.templates, param_combo):
            template_copy = copy.deepcopy(template)
            template_copy.update(params)
            current_params.append(template_copy)

        # Run experiment
        try:
            self.main_func(*current_params, exp_name)
            print(f"[Process {current_process().name}] Experiment {exp_name} completed successfully")
        except Exception as e:
            print(f"[Process {current_process().name}] Error in experiment {exp_name}: {str(e)}")
            print(traceback.format_exc())
            raise

    def run(self):
        """Run all experiments (supports parallel)"""
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
                        # Cancel all unfinished tasks
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        future.result(timeout=60)  # Add timeout limit
                    except concurrent.futures.TimeoutError:
                        print("A task timed out")
                    except Exception as e:
                        print(f"Task failed with: {e}")
                        # Write error info to error.txt
                        with open("error.txt", "a") as f:
                            f.write(f"Task failed with: {traceback.format_exc()}\n")

        except KeyboardInterrupt:
            print("\nCaught KeyboardInterrupt, shutting down...")
            return
        finally:
            # Ensure clean up all resources
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