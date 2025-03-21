import argparse
import random
import yaml
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI
from colorama import Fore, init
from abc import ABC, abstractmethod
import logging
import openai
import anthropic
import time
from typing import List, Dict

from .tools.utils import get_logger

logger = get_logger("query", "info")
init()
load_dotenv()

class ModelClient(ABC):
    @abstractmethod
    def chat_completion(self, messages, **kwargs):
        pass

class OpenAIClient(ModelClient):
    def __init__(self, api_key=None, base_url=None):
        # Allow setting base_url through parameters
        self.api_key = api_key
        self.base_url = base_url
        self.client_params = {"api_key": api_key}
        if base_url:
            self.client_params["base_url"] = base_url
        self.client = OpenAI(**self.client_params)
        self.error_count = 0
        self.error_threshold = 5

    def chat_completion(self, messages, **kwargs):
        while True:
            try:
                completion = self.client.chat.completions.create(
                    messages=messages,
                    **kwargs
                )
                self.error_count = 0  # Reset error count after success
                ret = completion.choices[0].message.content
                if ret == None:
                    logger.warning("Failed to get a response from the model. Details:")
                    logger.warning(completion)
                return ret
            except openai.RateLimitError as e:
                # Wait a short time and retry when rate limit is exceeded
                sleep_time = random.uniform(5, 10)
                logger.warning(f"Credit limit exceeded, retrying in {sleep_time} seconds... Error: {e}")
                time.sleep(sleep_time)
            except openai.APITimeoutError as e:
                # API timeout, wait and retry
                logger.warning(f"API timeout, retrying in 3 seconds... Error: {e}")
                time.sleep(3)
            except openai.InternalServerError as e:
                # Usually a custom error, wait and retry
                self.error_count += 1
                logger.warning(f"Internal server error ({self.error_count}/{self.error_threshold}), retrying... Error: {e}")
                
                if self.error_count >= self.error_threshold:
                    logger.warning("Too many errors, recreating client...")
                    self.client.close()
                    self.client = OpenAI(**self.client_params)
                    self.error_count = 0
                
                time.sleep(random.randint(20, 40))
            except openai.APIStatusError as e:
                logger.warning(f"Error during OpenAI API request: {e}")
                return str(e)

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
                    raise e
            except openai.BadRequestError as e:
                logger.warning(f"Bad request error: {e}")
                return str(e)

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

class AnthropicClient(ModelClient):
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)

    def chat_completion(self, messages: List[Dict], **kwargs):
        # Move the system prompt out of the messages list
        if len(messages) > 0 and messages[0]['role'] == 'system':
            system = messages.pop(0)['content']

        kwargs.pop('frequency_penalty')  # Remove model from kwargs

        completion = self.client.messages.create(
            system=system,
            messages=messages,
            model=kwargs['model'],
            stop_sequences=kwargs['stop'],
            max_tokens=kwargs['max_tokens'],
            temperature=kwargs['temperature'],
            top_p=kwargs['top_p'],
        )
        return completion.content[0].text

class QueryProcessor:
    def __init__(self, query, full_output=False):
        self.query = query
        self.client = self._get_client()
        self.full_output = full_output

        # Set logger level based on full_output
        if not self.full_output:
            logger.setLevel(logging.WARNING)  # Only output warnings and errors

    def _get_client(self):
        model_type = self.query["model_type"]
        if model_type == "openai":
            # Get base_url from configuration
            base_url = self.query.get("base_url")
            return OpenAIClient(
                api_key=get_api_key("openai"),
                base_url=base_url
            )
        elif model_type == "aiml":
            return AIMLClient(api_key=get_api_key("aiml"))
        elif model_type == "local":
            server = read_yaml(self.query['model'])
            base_url = f"http://{server.get('host', '127.0.0.1')}:{server.get('port', 8000)}/v1"
            self.query['model'] = server['model-tag']
            return LocalClient(base_url)
        elif model_type == "infinigence":
            return InfinigenceClient(api_key=get_api_key("infinigence"))
        elif model_type == "anthropic":
            return AnthropicClient(api_key=get_api_key("anthropic"))
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def process_query(self):
        if self.query['query_type'] == 'chat':
            return self.process_chats()
        elif self.query['query_type'] == 'instruction':
            return self.process_instruction()
        else:
            raise ValueError(f"Unsupported query type: {self.query['query_type']}")

    def _get_stop_pattern(self, model_name):
        # Set default stop pattern based on model name
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
            # For other models, use a generic stop pattern
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
                if response == None:
                    return "[FAILED] The model returned None."
                return response

            for message in messages:
                if message['role'] == 'assistant' and message['content'] == '***TBA***':
                    # If it's a TBA message from assistant, send previous context to LLM
                    response = send_to_llm(updated_messages)
                    updated_messages.append({"role": "assistant", "content": response})
                    llm_responses.append(response)
                    if self.full_output:
                        print(f"Assistant: " + Fore.YELLOW + f"{response}" + Fore.RESET)
                else:
                    updated_messages.append(message)
                    if self.full_output:
                        print(f"{message['role'].capitalize()}: {message['content']}")

            # Flush messages if the last message is not from the assistant
            if updated_messages[-1]['role'] != 'assistant':
                response = send_to_llm(updated_messages)
                updated_messages.append({"role": "assistant", "content": response})
                llm_responses.append(response)
                if self.full_output:
                    print(f"Assistant: " + Fore.YELLOW + f"{response}" + Fore.RESET)

            all_responses.append(llm_responses)

        return all_responses

    def process_instruction(self) -> list:
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
        
    def query(self, prompt: List[Dict[str, str]], chat_name: str, query_hook_dict: dict = None) -> str:
        # Available hooks:
        # pre_hook: function(prompt: List[Dict[str, str]]) -> Dict
        # post_hook: function(llm_response: str) -> Dict

        config = self.query_config.copy()
        
        pre_hook_warning = None
        
        if query_hook_dict and 'pre_hook' in query_hook_dict:
            pre_hook_result = query_hook_dict['pre_hook'](prompt)
            
            # Check if pre_hook returned a dictionary with the expected format
            if isinstance(pre_hook_result, dict) and "result" in pre_hook_result:
                # Store warning info if status is "warning"
                if pre_hook_result.get("status") == "warning":
                    pre_hook_warning = pre_hook_result.get("info")
                    
                # If status is "filtered", return the info message directly
                if pre_hook_result.get("status") == "filtered":
                    return [pre_hook_result.get("info", "Query rejected by security filter")]
                    
                prompt = pre_hook_result["result"]
            else:
                # For backward compatibility
                prompt = pre_hook_result
            
        # If prompt is None, return early
        if prompt is None:
            return ["Query rejected by security filter"]
        
        chat_config = {
            "name": chat_name,
            "messages": prompt
        }
        config['chats'] = [chat_config]
        llm_response = QueryProcessor(config).process_query()[0]
        
        # Append warning to the response if there was one
        if pre_hook_warning:
            llm_response = f"[WARNING] {pre_hook_warning}\n\n{llm_response}"
        
        if query_hook_dict and 'post_hook' in query_hook_dict:
            post_hook_result = query_hook_dict['post_hook'](llm_response)
            
            # Check if post_hook returned a dictionary with the expected format
            if isinstance(post_hook_result, dict) and "result" in post_hook_result:
                llm_response = post_hook_result["result"]
            else:
                # For backward compatibility
                llm_response = post_hook_result

        return llm_response

if __name__ == "__main__":
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