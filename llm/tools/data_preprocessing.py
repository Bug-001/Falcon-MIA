import os
import pickle
import functools
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

def cache_to_disk(func):
    @functools.wraps(func)
    def wrapper_cache(*args, **kwargs):
        data_config = args[0]  # 假设第一个参数是 data_config
        cache_enabled = getattr(data_config, 'cache_enabled', False)
        if not cache_enabled:
            return func(*args, **kwargs)
        
        cache_dir = getattr(data_config, 'preprocessed_cache_dir', None)
        if cache_dir is None:
            logger.warning("Cache directory not provided. Skipping caching.")
            return func(*args, **kwargs)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        func_name = func.__name__.replace("/", "")
        cache_file = os.path.join(cache_dir, f"{func_name}.pkl")

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                logger.info(f"Loading preprocessed cached data for {func.__name__}")
                return pickle.load(f)

        result = func(*args, **kwargs)

        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
            logger.info(f"Cached preprocessed data for {func.__name__}")
        return result

    return wrapper_cache

class DefaultProcessor:
    def __init__(self, data_config):
        self.data_config = data_config
        self.input = "Answer the following question:\n\nQuestion: {text}\n\nAnswer:"
        self.label = "{label}"
        self.dataset_name = data_config.dataset_name
    
    @cache_to_disk
    def process(self):
        dataset = load_dataset(self.dataset_name, cache_dir=self.data_config.hf_cache_dir)  # 使用HuggingFace缓存目录
        dataset = dataset.map(self.apply_prompt)
        
        train_set = dataset["train"]
        test_set = dataset["test"]
        
        return train_set, test_set, test_set

    def get_prompt(self):
        prompt_template = self.data_config.prompt_template
        if prompt_template == 'custom':
            return self.data_config.prompt_input
        elif prompt_template == 'default':
            return self.input
        else:
            raise ValueError(f"Unsupported prompt template: {prompt_template}")

    def apply_prompt(self, example):
        input_prompt = self.get_prompt()
        output = self.label.format(**example)
        return {
            "x": input_prompt.format(**example),
            "y": self.label_map.get(output, output)
        }

class AGNewsProcessor(DefaultProcessor):
    def __init__(self, data_config):
        super().__init__(data_config)
        self.input = "Classify the following news article into one of these categories: World, Sports, Business, Sci/Tech.\n\nArticle: {text}\n\nCategory:"
        self.label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

class TRECProcessor(DefaultProcessor):
    def __init__(self, data_config):
        super().__init__(data_config)
        self.input_prompt = "Classify the following question based on whether its answer type is a Number, Location, Person, Description, Entity, or Abbreviation.\n\nQuestion: {text}\n\nAnswer:"
        self.labels = "{coarse_label}"
        self.label_map = {
            0: "Abbreviation",
            2: "Description",
            1: "Entity",
            3: "Person",
            4: "Location",
            5: "Number"
        }

def preprocess(data_config):
    dataset_processors = {
        "ag_news": AGNewsProcessor,
        # Add more dataset processors here
    }
    
    processor_class = dataset_processors.get(data_config.dataset_name)
    if not processor_class:
        raise ValueError(f"Unsupported dataset: {data_config.dataset_name}")
    
    processor = processor_class(data_config)
    train_dataset, valid_dataset, test_dataset = processor.process()
    
    # 应用数据集范围选择
    if hasattr(data_config, 'train') and hasattr(data_config.train, 'end_index') and data_config.train.end_index > 0:
        train_dataset = train_dataset.select(range(data_config.train.start_index, data_config.train.end_index))
    if hasattr(data_config, 'eval') and hasattr(data_config.eval, 'end_index') and data_config.eval.end_index > 0:
        valid_dataset = valid_dataset.select(range(data_config.eval.start_index, data_config.eval.end_index))
    
    return train_dataset, valid_dataset, test_dataset

# Example usage and testing
if __name__ == "__main__":
    class DummyConfig:
        dataset_name = "ag_news"
        prompt_template = "default"
        prompt_input = "Custom prompt: {text}\nCategory:"
        cache_enabled = False
        train = type('obj', (object,), {'start_index': 0, 'end_index': 1000})
        eval = type('obj', (object,), {'start_index': 0, 'end_index': 100})

        def get(self, key, default=None):
            return getattr(self, key, default)
    
    config = DummyConfig()
    train, valid, test = preprocess(config)
    print(f"Train size: {len(train)}")
    print(f"Valid size: {len(valid)}")
    print(f"Test size: {len(test)}")
    print(f"Sample: {train[0]}")