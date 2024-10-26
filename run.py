import argparse
import yaml
import itertools
from icl import main as icl_main
from dotenv import load_dotenv

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def generate_param_combinations(params):
    param_names = list(params.keys())
    param_values = list(params.values())
    combinations = list(itertools.product(*param_values))
    return [dict(zip(param_names, combo)) for combo in combinations]

def run_experiments(data_config, attack_config, query_config, test_params):
    param_combinations = generate_param_combinations(test_params['data'])
    
    for params in param_combinations:
        current_data_config = data_config.copy()
        current_data_config.update(params)
        
        current_attack_config = attack_config.copy()
        current_attack_config['name'] = f"exp_" + "_".join([f"{k}_{v}" for k, v in params.items()])
        
        print(f"Running experiment: {current_attack_config['name']}")
        icl_main(current_data_config, current_attack_config, query_config)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data.yaml', help='Path to the data config file')
    parser.add_argument('--attack', default='attack_chat.yaml', help='Path to the attack config file')
    parser.add_argument('--query', default='query.yaml', help='Path to the query config file')
    parser.add_argument('--test_params', default='params.yaml', help='Path to the test parameters config file')
    args = parser.parse_args()

    data_config = load_yaml_config(args.data)
    attack_config = load_yaml_config(args.attack)
    query_config = load_yaml_config(args.query)
    test_params = load_yaml_config(args.test_params)

    run_experiments(data_config, attack_config, query_config, test_params)

if __name__ == "__main__":
    load_dotenv()
    main()