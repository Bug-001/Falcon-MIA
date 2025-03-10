import argparse
import yaml
import subprocess
import os

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def select_chat_template(model_name):
    # Automatically get all template files in the templates directory
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    default_template = 'general.jinja'
    
    # Iterate through all .jinja files in the templates directory, looking for matches
    for template_file in os.listdir(template_dir):
        if template_file.endswith('.jinja'):
            # Extract model name from filename (remove .jinja suffix)
            template_model = template_file[:-6]
            if template_model in model_name.lower():
                return os.path.join(os.path.dirname(__file__), 'templates', template_file)
    
    # If no matching template is found, return the default template
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

    # Select appropriate chat template
    chat_template = select_chat_template(model_name)
    # if chat_template.endswith('mistral.jinja'):
    #     vllm_command.extend(["--tokenizer_mode", "mistral"])
    vllm_command.extend(["--chat-template", chat_template])

    print("Executing command:", " ".join(vllm_command))
    subprocess.run(vllm_command)

if __name__ == "__main__":
    main()