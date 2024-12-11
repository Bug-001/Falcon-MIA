import argparse
import yaml
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI
from colorama import Fore, init
from abc import ABC, abstractmethod
import logging
import openai
import time
from typing import List, Dict

from .tools.utils import get_logger

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
        while True:
            try:
                completion = self.client.chat.completions.create(
                    messages=messages,
                    **kwargs
                )
                return completion.choices[0].message.content
            except openai.InternalServerError as e:
                if e.status_code == 502:
                    print("Server is not connected. Retrying in 30 seconds...")
                    time.sleep(30)
                else:
                    raise

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

class AIMLClient(ModelClient):
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, base_url="https://api.aimlapi.com/v1")

    def chat_completion(self, messages, **kwargs):
        while True:
            try:
                completion = self.client.chat.completions.create(
                    messages=messages,
                    **kwargs
                )
                return completion.choices[0].message.content
            except openai.InternalServerError as e:
                time.sleep(5)
            except Exception as e:
                logger.warning(f"Error during AI/ML API request: {type(e)}")
                raise e
                return None

class QueryProcessor:
    def __init__(self, query, full_output=False):
        self.query = query
        self.client = self._get_client()
        self.full_output = full_output

        # 根据 full_output 设置 logger 级别
        if not self.full_output:
            logger.setLevel(logging.WARNING)  # 只输出警告和错误

    def _get_client(self):
        model_type = self.query["model_type"]
        if model_type == "openai":
            return OpenAIClient(api_key=get_api_key("openai"))
        elif model_type == "aiml":
            return AIMLClient(api_key=get_api_key("aiml"))
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
        ret = None
        while ret == None:
            if self.query['query_type'] == 'chat':
                ret = self.process_chats()
            elif self.query['query_type'] == 'instruction':
                ret = self.process_instruction()
            else:
                raise ValueError(f"Unsupported query type: {self.query['query_type']}")
        return ret

    def _get_stop_pattern(self, model_name):
        # 根据模型名称设置默认的stop pattern
        model_name_lower = model_name.lower()
        if 'llama' in model_name_lower:
            return ["[INST]", "[/INST]", "</s>", "[/s]"]
        elif 'claude' in model_name_lower:
            return ["\n\nHuman:", "\n\nAssistant:"]
        elif 'mistral' in model_name_lower:
            return ["[INST]", "[/INST]"]
        elif 'vicuna' in model_name_lower:
            return ["[INST]", "[/INST]", "</s>", "[/s]"]
        else:
            # 对于其他模型,使用一个通用的stop pattern
            return []

    def process_chats(self):
        chats = self.query.get('chats', [])

        all_responses = []

        for i, chat in enumerate(chats):
            logger.info(f"Processing chat: {chat.get('name', 'Unnamed chat')}")
            messages = chat.get('messages', [])
            updated_messages = []
            
            model_name = self.query.get('model', 'gpt-3.5-turbo')
            stop_pattern = self._get_stop_pattern(model_name)

            llm_responses = []  # To store LLM responses

            def send_to_llm(messages):
                response = self.client.chat_completion(
                    messages,
                    model=model_name,
                    temperature=self.query.get('temperature', 0.7),
                    max_tokens=self.query.get('max_tokens', 150),
                    top_p=self.query.get('top_p', 1.0),
                    frequency_penalty=self.query.get('frequency_penalty', 0.0),
                    presence_penalty=self.query.get('presence_penalty', 0.0),
                    n=self.query.get('n', 1),
                    stop=self.query.get('stop', []) + stop_pattern,
                )
                if response is not None:
                    return response
                else:
                    logger.warning("Failed to get a response from the model.")
                    return None

            for message in messages:
                if message['role'] == 'assistant' and message['content'] == '***TBA***':
                    # 如果是assistant的TBA消息,则将前述上文发送到LLM
                    response = send_to_llm(updated_messages)
                    if response:
                        updated_messages.append({"role": "assistant", "content": response})
                        llm_responses.append(response)
                        if self.full_output:
                            print(f"Assistant: " + Fore.YELLOW + f"{response}".strip() + Fore.RESET)
                    else:
                        logger.warning("Failed to get a response from the model.")
                else:
                    updated_messages.append(message)
                    if self.full_output:
                        print(f"{message['role'].capitalize()}: {message['content']}".strip())

            # Flush messages if the last message is not from the assistant
            if updated_messages[-1]['role'] != 'assistant':
                response = send_to_llm(updated_messages)
                if response is not None:
                    updated_messages.append({"role": "assistant", "content": response})
                    llm_responses.append(response)
                    if self.full_output:
                        print(f"Assistant: " + Fore.YELLOW + f"{response}".strip() + Fore.RESET)
                else:
                    logger.warning("Failed to get a response from the model.")

            all_responses.append(llm_responses)

        return all_responses

    def process_instruction(self):
        # TODO: Implement instruction processing
        raise NotImplementedError("Instruction query type is not yet supported.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Large Language Model Query Tool")
    parser.add_argument("-i", "--interactive", action="store_true", 
                        help="Run in interactive mode")
    parser.add_argument("-f", "--full", action="store_true", 
                        help="Output full conversation (default: LLM responses only)")
    parser.add_argument("-c", "--config", default=os.path.join(os.path.dirname(__file__), "examples", "query.yaml"), 
                        help="Path to the YAML query file (default: query.yaml)")
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

class ModelInterface:
    def __init__(self, query_config):
        self.query_config = query_config
        
    def query(self, prompt: List[Dict[str, str]], chat_name: str) -> str:
        config = self.query_config.copy()
        chat_config = {
            "name": chat_name,
            "messages": prompt
        }
        config['chats'] = [chat_config]
        
        llm_response = QueryProcessor(config).process_query()[0]
        return llm_response

if __name__ == "__main__":
    load_dotenv()
    args = parse_arguments()
    query = read_yaml(args.config)

    processor = QueryProcessor(query, args.full)
    
    if args.interactive:
        logger.info("Running in interactive mode.")
        # TODO: Implement interactive mode logic
    else:
        logger.info("Running in non-interactive mode.")
        print('----------')
        for s in processor.process_query()[0]:
            print(s, end='\n----------\n')