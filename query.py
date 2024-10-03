import argparse
import yaml
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI

def parse_arguments():
    parser = argparse.ArgumentParser(description="Large Language Model Query Tool")
    parser.add_argument("-i", "--interactive", action="store_true", 
                        help="Run in interactive mode")
    parser.add_argument("-q", "--query_file", default="examples/query.yaml", 
                        help="Path to the YAML query file (default: query.yaml)")
    parser.add_argument("-p", "--prompt_file", default="examples/prompt-templates.yaml",
                        help="Path to the prompt templates file (default: prompt_templates.yaml)")
    return parser.parse_args()

def get_api_key(provider):
    """
    根据提供商名称获取API密钥
    """
    key_name = f"{provider.upper()}_API_KEY"
    api_key = os.getenv(key_name)
    
    if not api_key:
        raise ValueError(f"API key for {provider} not found. Please set the {key_name} environment variable in .env.")
    
    return api_key

def format_prompt(template, **kwargs):
    return template.format(**kwargs)

def openai_api_request(client, query):
    try:
        messages = [
            {"role": "system", "content": query.get('system_message', "You are a helpful assistant.")},
            {"role": "user", "content": query['instruction']}
        ]

        # Prepare the completion parameters
        completion_params = {
            'model': query.get('model', 'gpt-3.5-turbo'),
            'messages': messages,
            'temperature': query.get('temperature', 0.7),
            'max_tokens': query.get('max_tokens', 150),
            'top_p': query.get('top_p', 1.0),
            'frequency_penalty': query.get('frequency_penalty', 0.0),
            'presence_penalty': query.get('presence_penalty', 0.0),
            'n': query.get('n', 1)
        }

        # Add optional parameters only if they are present in the query
        if 'stop' in query:
            completion_params['stop'] = query['stop']

        completion = client.chat.completions.create(**completion_params)
        print(completion.choices[0].message)
        return completion.choices[0].message
    except Exception as e:
        print(f"Error during API request: {e}")
        return None

def model_query_local(query):
    server = read_yaml(query['model'])
    server_host = server.get('host', '127.0.0.1')
    server_port = server.get('port', 8000)
    client = OpenAI(
        base_url=f"http://{server_host}:{server_port}/v1",
    )
    if server_host == 'localhost' or server_host.startswith('127.'):
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)

    query['model'] = server['model-tag']
    openai_api_request(client, query)

def model_query_openai(query):
    client = OpenAI()
    openai_api_request(client, query)

def model_query_anthropic(query):
    url = "https://api.anthropic.com/maas/v1/engines/davinci/completions"

    payload = {
        "model": "davinci",
        "prompt": query,
        "max_tokens": 150
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer %s" % get_api_key("anthropic")
    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.json())

def model_query_infinigence(query):
    model = query["model"]

    url = f"https://cloud.infini-ai.com/maas/{model}/nvidia/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": query['formatted_prompt']
            },
        ]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer %s" % get_api_key("infinigence")
    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.json())

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def format_prompt(template, **kwargs):
    return template.format(**kwargs)

def process_query(query, prompt_templates):
    prompt_type = query.get('prompt_type', 'default')
    template = prompt_templates.get(prompt_type, prompt_templates['default'])
    
    # 准备格式化参数
    format_args = {k: v for k, v in query.items() if k != 'prompt_type'}
    
    # 格式化prompt
    formatted_prompt = format_prompt(template['template'], **format_args)
    
    # 更新query字典
    query['formatted_prompt'] = formatted_prompt

def dispatch_query(query):
    if query["type"] == "openai":
        # 这里实现 OpenAI 模型的查询逻辑
        model_query_openai(query)
    elif query["type"] == "local":
        # 这里实现本地模型的查询逻辑
        model_query_local(query)
    elif query["type"] == "anthropic":
        # 这里实现 Anthropic 模型的查询逻辑
        pass
    elif query["type"] == "infinigence":
        # 这里实现 Infinigence 模型的查询逻辑
        model_query_infinigence(query)
    else:
        raise ValueError(f"Unknown type: {query['type']}")

def main():
    load_dotenv()
    args = parse_arguments()
    query = read_yaml(args.query_file)
    prompt_templates = read_yaml(args.prompt_file)

    # 处理query，应用prompt模板
    process_query(query, prompt_templates)
    
    if args.interactive:
        print("Running in interactive mode.")
        # 这里实现交互模式的逻辑
    else:
        print("Running in non-interactive mode.")
        dispatch_query(query)

if __name__ == "__main__":
    main()