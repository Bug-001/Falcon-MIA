import argparse
import yaml
import subprocess
import sys

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="VLLM server launcher")
    parser.add_argument("-c", "--config", type=str, help="Path to the YAML config file")
    
    args, unknown = parser.parse_known_args()

    vllm_command = []

    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if key == "model-tag":
                vllm_command = ["vllm", "serve", value] + vllm_command
            elif isinstance(value, bool):
                if value:
                    vllm_command.append(f"--{key}")
            else:
                vllm_command.extend([f"--{key}", str(value)])

    vllm_command.extend(unknown)
    vllm_command.extend(["--chat-template", "templates/template_llava.jinja"])

    print("Executing command:", " ".join(vllm_command))
    # todo: chat_template
    subprocess.run(vllm_command)

if __name__ == "__main__":
    main()