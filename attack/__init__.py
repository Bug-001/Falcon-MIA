from .common import *

from datasets import Dataset, DatasetDict, concatenate_datasets
from abc import ABC, abstractmethod
from functools import wraps

from llm.tools.utils import get_logger
from llm.query import ModelInterface
from utils import SLogger, SDatasetManager

logger = get_logger("ICL Attack", "info")

class ICLDataLoader:
    def __init__(self, dataset: DatasetDict, batch_size: int, train_num: int, test_num: int, selected_attack_sample: int = 0):
        self.train_dataset = dataset['train']
        self.valid_dataset = dataset['validation']
        self.test_dataset = dataset['test']
        self.batch_size = batch_size
        self.train_num = train_num
        self.test_num = test_num
        self.selected_attack_sample = selected_attack_sample
        self.atk_index = 0

    def _generate_data_alt(self, dataset, batch_num):
        # Alternately select member samples to ensure each attack instance has both member and non-member cases
        data = []
        icl_index = 0
        batch_num = batch_num // 2
        for i in range(batch_num):
            self.atk_index = (self.atk_index + 1) % len(self.valid_dataset)
            attack_sample = self.valid_dataset.select([self.atk_index])

            # Determine position to replace
            replace_position = i % self.batch_size

            # Make this sample a member
            is_member = True
            icl_samples = self._get_batch(dataset, icl_index)
            icl_index = (icl_index + self.batch_size) % len(dataset)
            # Replace specified position in icl_samples with attack_sample
            icl_samples_dict = icl_samples.to_dict()
            for key in icl_samples_dict.keys():
                icl_samples_dict[key][replace_position] = attack_sample[key][0]
            icl_samples = Dataset.from_dict(icl_samples_dict, features=icl_samples.features)
            attack_sample = attack_sample[0]
            data.append((icl_samples, attack_sample, is_member))
            
            # Make this sample a non-member
            is_member = False
            icl_samples = self._get_batch(dataset, icl_index)
            icl_index = (icl_index + self.batch_size) % len(dataset)
            data.append((icl_samples, attack_sample, is_member))

        return data

    def _generate_data(self, dataset, batch_num):
        if self.selected_attack_sample == 'alt':
            return self._generate_data_alt(dataset, batch_num)

        data = []
        icl_index = 0
        test_index = 0
        for i in range(batch_num):
            icl_samples = self._get_batch(dataset, icl_index)
            icl_index = (icl_index + self.batch_size) % len(dataset)

            is_member = (i % 2 == 0)
            if is_member:
                if self.selected_attack_sample == 0:
                    # Pseudo-random selection strategy
                    attack_sample = icl_samples[i % self.batch_size]
                elif 1 <= self.selected_attack_sample <= self.batch_size:
                    # Select member sample at specified index
                    attack_sample = icl_samples[self.selected_attack_sample - 1]
                else:
                    raise ValueError(f"Invalid selected_attack_sample: {self.selected_attack_sample}")
            else:
                # Select non-member sample
                attack_sample = self.valid_dataset[test_index]
                test_index = (test_index + 1) % len(self.valid_dataset)

            data.append((icl_samples, attack_sample, is_member))
        return data

    def _get_batch(self, dataset: Dataset, start_index) -> Dataset:
        end_index = start_index + self.batch_size
        if end_index <= len(dataset):
            return dataset.select(range(start_index, end_index))
        else:
            # Handle wrap around case
            first_part = list(range(start_index, len(dataset)))
            second_part = list(range(0, end_index % len(dataset)))
            return dataset.select(first_part + second_part)

    def train(self):
        return self._generate_data(self.train_dataset, self.train_num)
    
    def test(self):
        return self._generate_data(self.test_dataset, self.test_num)

class ICLAttackStrategy(ABC):
    def __init__(self, attack_config: Dict[str, Any]):
        self.attack_config = attack_config
        self.random_seed = attack_config.get('random_seed', random.randint(0, 1000000))
        random.seed(self.random_seed)
        self.results = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = SLogger(attack_config.get('name'))

    def prepare(self, data_config: Dict[str, Any], data_loader: ICLDataLoader = None):
        dataset_name = data_config["dataset"]
        task = data_config.get("task", "default")
        
        # Load dataset and its default configuration
        self.sdm = SDatasetManager(dataset_name, task)
        train_attack = self.attack_config.get('train_attack', 100)
        test_attack = self.attack_config.get('test_attack', 100)
        num_demo = data_config.get('num_demonstrations', 6)
        # Verify dataset quantities, requiring validation set size not to exceed about 3/4 of total data
        total_size = self.sdm.get_total_size()
        if train_attack + test_attack > total_size * 3 // 4:
            train_attack = int(train_attack / (train_attack + test_attack) * total_size * 3 // 4)
            test_attack = total_size * 3 // 4 - train_attack
        self.train_attack = train_attack
        self.test_attack = test_attack
        num_valid = train_attack + test_attack
        num_train = train_attack * num_demo
        num_test = test_attack * num_demo
        split = [num_train, num_valid, num_test]
        dataset = self.sdm.crop_dataset(split=split, seed=self.random_seed, prioritized_splits=['validation'])
        default_config = self.sdm.get_config()

        # Merge default configuration and user configuration
        self.data_config = {
            **default_config,
            **data_config
        }
        if data_loader:
            self.data_loader = data_loader
        else:
            self.data_loader = ICLDataLoader(
                dataset=dataset,
                batch_size=self.data_config["num_demonstrations"],
                train_num=train_attack,
                test_num=test_attack,
                selected_attack_sample=self.attack_config.get('selected_attack_sample', 0)
            )

        # Select different templates for training and testing
        templates = self.data_config['prompt_template']
        self.test_template = templates[-1]  # Test always uses the last template
        self.train_template = templates[-2]  # Training always uses the second last template

    def remove_punctuation(self, word: str):
        return word.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    
    def generate_prompt(self, user_template, assistant_template, samples):
        '''
        Generate a list of prompts from the given messages, like 
        [{"input": "hello!", "output": "hello! what can I help you today?"}, ...].
        '''    
        ret = []
        for sample in samples:
            ret.append([{
                "role": "user",
                "content": user_template.format(input=sample["input"])
            }, {
                "role": "assistant",
                "content": assistant_template.format(output=sample["output"])
            }])
        return ret

    def generate_icl_prompt(self, icl_samples: Dataset, is_train: bool = True):
        template = self.train_template if is_train else self.test_template
        
        prompt = [{
            "role": "system",
            "content": template['system']
        }]

        for sample in icl_samples:
            prompt.append({
                "role": "user",
                "content": template['user'].format(input=sample["input"])
            })
            prompt.append({
                "role": "assistant",
                "content": template['assistant'].format(output=sample["output"])
            })

        return prompt

    def get_results_filename(self):
        return f"{self.__class__.__name__}_results.json"

    def save_results(self):
        class CustomJSONizer(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                return json.JSONEncoder.default(self, obj)
        filename = self.get_results_filename()
        with open(filename, 'w') as f:
            json.dump(self.results, f, cls=CustomJSONizer)
        logger.info(f"Results saved to {filename}")

    def load_results(self):
        filename = self.get_results_filename()
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.results = json.load(f)
            logger.info(f"Results loaded from {filename}")
            return True
        return False

    @staticmethod
    def cache_results(attack_method):
        @wraps(attack_method)
        def wrapper(self, model: 'ModelInterface'):
            # if self.load_results():
            #     logger.info("Loaded previous results. Skipping attack.")
            #     return
            attack_method(self, model)
            # self.save_results()
        return wrapper

    @abstractmethod
    @cache_results
    def attack(self, model: 'ModelInterface'):
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        pass

    @staticmethod
    def create(attack_config: Dict[str, Any]) -> 'ICLAttackStrategy':
        from .gap import GAPAttack
        from .inquiry import InquiryAttack
        from .repeat import RepeatAttack
        from .brainwash import BrainwashAttack
        from .hybrid import HybridAttack
        from .obfuscation import ObfuscationAttack

        attack_type = attack_config['type']
        if attack_type == 'GAP':
            return GAPAttack(attack_config)
        elif attack_type == 'Inquiry':
            return InquiryAttack(attack_config)
        elif attack_type == 'Repeat':
            return RepeatAttack(attack_config)
        elif attack_type == 'Brainwash':
            return BrainwashAttack(attack_config)
        elif attack_type == 'Hybrid':
            return HybridAttack(attack_config)
        elif attack_type == "Obfuscation":
            return ObfuscationAttack(attack_config)
        else:
            return None
