import argparse
import yaml
import requests
import os
from dotenv import load_dotenv

def parse_arguments():
    parser = argparse.ArgumentParser(description="Large Language Model Query Tool")
    parser.add_argument("-i", "--interactive", action="store_true", 
                        help="Run in interactive mode")
    parser.add_argument("-q", "--query_file", default="query.yaml", 
                        help="Path to the YAML query file (default: query.yaml)")
    parser.add_argument("-p", "--prompt_file", default="prompt_templates.yaml",
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

def model_query_local(query):
    # 这里实现本地模型的查询逻辑
    pass

def model_query_openai(query):
    url = "https://api.openai.com/v1/engines/davinci/completions"

    payload = {
        "model": "davinci",
        "prompt": query,
        "max_tokens": 150
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer %s" % get_api_key("openai")
    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.json())

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
                "role": "system",
                "content": "第一个问题：你是谁？"
            },
            {
                "role": "system",
                "content": "第二个问题：我是谁？"
            }
        ]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer %s" % get_api_key("infinigence")
    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.json())

def read_config(query_file):
    with open(query_file, 'r') as file:
        return yaml.safe_load(file)

def dispatch_query(query):
    if query["type"] == "openai":
        # 这里实现 OpenAI 模型的查询逻辑
        pass
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
    query = read_config(args.query_file)
    
    if args.interactive:
        print("Running in interactive mode.")
        # 这里实现交互模式的逻辑
    else:
        print("Running in non-interactive mode.")
        dispatch_query(query)

if __name__ == "__main__":
    main()