import argparse
import yaml
import subprocess
import os

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def select_chat_template(model_name):
    # 定义模型名称与模板文件的映射
    template_mapping = {
        'llama': 'llama.jinja',
        'claude': 'claude.jinja',
        'mistral': 'mistral.jinja',
        'vicuna': 'llama.jinja',
        'gpt2-xl': 'gpt2-xl.jinja',
        'qwen': 'qwen.jinja',
        # 可以根据需要添加更多映射
    }
    
    # 默认模板
    default_template = 'general.jinja'
    
    # 查找匹配的模板
    for key, template in template_mapping.items():
        if key in model_name.lower():
            return os.path.join(os.path.dirname(__file__), 'templates', template)
    
    # 如果没有找到匹配的模板，返回默认模板
    return os.path.join(os.path.dirname(__file__), 'templates', default_template)

def main():
    parser = argparse.ArgumentParser(description="VLLM server launcher")
    parser.add_argument("-c", "--config", type=str, help="Path to the YAML config file")
    
    args, unknown = parser.parse_known_args()

    vllm_command = []
    model_name = ""

    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if key == "model-tag":
                model_name = value
                vllm_command = ["vllm", "serve", value] + vllm_command
            elif isinstance(value, bool):
                if value:
                    vllm_command.append(f"--{key}")
            else:
                vllm_command.extend([f"--{key}", str(value)])

    vllm_command.extend(unknown)

    # 选择合适的chat template
    chat_template = select_chat_template(model_name)
    # if chat_template.endswith('mistral.jinja'):
    #     vllm_command.extend(["--tokenizer_mode", "mistral"])
    vllm_command.extend(["--chat-template", chat_template])

    print("Executing command:", " ".join(vllm_command))
    subprocess.run(vllm_command)

if __name__ == "__main__":
    main()