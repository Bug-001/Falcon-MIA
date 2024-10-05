import argparse
import yaml
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI
from tools.utils import get_logger
from colorama import Fore, init
from abc import ABC, abstractmethod
import logging

logger = get_logger("query", "info")
init()

class ModelClient(ABC):
    @abstractmethod
    def chat_completion(self, messages, **kwargs):
        pass

class OpenAIClient(ModelClient):
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key)

    def chat_completion(self, messages, **kwargs):
        try:
            completion = self.client.chat.completions.create(
                messages=messages,
                **kwargs
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.warning(f"Error during OpenAI API request: {e}")
            return None

class LocalClient(ModelClient):
    def __init__(self, base_url):
        self.client = OpenAI(base_url=base_url)

    def chat_completion(self, messages, **kwargs):
        try:
            completion = self.client.chat.completions.create(
                messages=messages,
                **kwargs
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.warning(f"Error during Local API request: {e}")
            return None

class InfinigenceClient(ModelClient):
    def __init__(self, api_key):
        self.api_key = api_key

    def chat_completion(self, messages, **kwargs):
        model = kwargs.get('model', 'qwen2.5-7b-instruct')
        url = f"https://cloud.infini-ai.com/maas/{model}/nvidia/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.warning(f"Error during Infinigence API request: {e}")
            return None

class QueryProcessor:
    def __init__(self, query, prompt_templates, full_output):
        self.query = query
        self.prompt_templates = prompt_templates
        self.client = self._get_client()
        self.full_output = full_output

        # 根据 full_output 设置 logger 级别
        if not self.full_output:
            logger.setLevel(logging.WARNING)  # 只输出警告和错误

    def _get_client(self):
        model_type = self.query["model_type"]
        if model_type == "openai":
            return OpenAIClient(api_key=get_api_key("openai"))
        elif model_type == "local":
            server = read_yaml(self.query['model'])
            base_url = f"http://{server.get('host', '127.0.0.1')}:{server.get('port', 8000)}/v1"
            self.query['model'] = server['model-tag']
            return LocalClient(base_url)
        elif model_type == "infinigence":
            return InfinigenceClient(api_key=get_api_key("infinigence"))
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def process_query(self):
        if self.query['query_type'] == 'chat':
            self.process_chats()
        elif self.query['query_type'] == 'instruction':
            self.process_instruction()
        else:
            raise ValueError(f"Unsupported query type: {self.query['query_type']}")

    def _get_stop_pattern(self, model_name):
        # 根据模型名称设置默认的stop pattern
        model_name_lower = model_name.lower()
        if 'llama' in model_name_lower:
            return ["Human:", "Assistant:"]
        elif 'claude' in model_name_lower:
            return ["\n\nHuman:", "\n\nAssistant:"]
        elif 'mistral' in model_name_lower:
            return ["[INST]", "[/INST]"]
        else:
            # 对于其他模型,使用一个通用的stop pattern
            return ["\nUser:", "\nAssistant:"]

    def process_chats(self):
        chats = self.query.get('chats', [])

        for i, chat in enumerate(chats):
            logger.info(f"Processing chat: {chat.get('name', 'Unnamed chat')}")
            messages = chat.get('messages', [])
            updated_messages = []
            
            model_name = self.query.get('model', 'gpt-3.5-turbo')
            stop_pattern = self._get_stop_pattern(model_name)

            llm_responses = []  # To store LLM responses

            for message in messages:
                if message['role'] == 'assistant' and message['content'] == '***TBA***':
                    response = self.client.chat_completion(
                        updated_messages,
                        model=model_name,
                        temperature=self.query.get('temperature', 0.7),
                        max_tokens=self.query.get('max_tokens', 150),
                        top_p=self.query.get('top_p', 1.0),
                        frequency_penalty=self.query.get('frequency_penalty', 0.0),
                        presence_penalty=self.query.get('presence_penalty', 0.0),
                        n=self.query.get('n', 1),
                        stop=self.query.get('stop', []) + stop_pattern,
                    )
                    if response:
                        message['content'] = response
                        updated_messages.append(message)
                        llm_responses.append(response)
                        if self.full_output:
                            print(f"{message['role'].capitalize()}: " + Fore.YELLOW + f"{message['content']}".strip() + Fore.RESET)
                    else:
                        logger.warning("Failed to get a response from the model.")
                        break
                else:
                    updated_messages.append(message)
                    if self.full_output:
                        print(f"{message['role'].capitalize()}: {message['content']}".strip())

            # 在每个chat结束后添加分隔符
            if not self.full_output:
                if llm_responses:  # 如果当前chat有LLM响应
                    print("\n-----\n".join(llm_responses))
                    if i < len(chats) - 1:  # 如果不是最后一个chat
                        print("\n----------\n")

    def process_instruction(self):
        # TODO: Implement instruction processing
        raise NotImplementedError("Instruction query type is not yet supported.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Large Language Model Query Tool")
    parser.add_argument("-i", "--interactive", action="store_true", 
                        help="Run in interactive mode")
    parser.add_argument("-f", "--full", action="store_true", 
                        help="Output full conversation (default: LLM responses only)")
    parser.add_argument("-q", "--query_file", default=os.path.join(os.path.dirname(__file__), "examples", "query.yaml"), 
                        help="Path to the YAML query file (default: query.yaml)")
    parser.add_argument("-p", "--prompt_file", default=os.path.join(os.path.dirname(__file__), "examples", "prompt-templates.yaml"),
                        help="Path to the prompt templates file (default: prompt_templates.yaml)")
    return parser.parse_args()

def get_api_key(provider):
    key_name = f"{provider.upper()}_API_KEY"
    api_key = os.getenv(key_name)
    
    if not api_key:
        raise ValueError(f"API key for {provider} not found. Please set the {key_name} environment variable in .env.")
    
    return api_key

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    load_dotenv()
    args = parse_arguments()
    query = read_yaml(args.query_file)
    prompt_templates = read_yaml(args.prompt_file)

    processor = QueryProcessor(query, prompt_templates, args.full)
    
    if args.interactive:
        logger.info("Running in interactive mode.")
        # TODO: Implement interactive mode logic
    else:
        logger.info("Running in non-interactive mode.")
        processor.process_query()

if __name__ == "__main__":
    main()