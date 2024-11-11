from typing import Dict, Tuple, Any, List
from datasets import load_dataset, Dataset
from abc import ABC, abstractmethod
import os

class BaseDataLoader(ABC):
    """数据集加载器的基类"""
    @abstractmethod
    def load(self, task: str = "default") -> Tuple[Dataset, Dict[str, Any]]:
        """
        加载并处理数据集
        Args:
            task: 任务名称，默认为"default"表示单任务数据集
        Returns:
            处理后的数据集和配置的元组
        """
        pass

    @abstractmethod
    def get_supported_tasks(self) -> List[str]:
        """返回该数据集支持的所有任务列表"""
        pass

class SQuADLoader(BaseDataLoader):
    def get_supported_tasks(self) -> List[str]:
        return ["qa", "verification", "generation"]
    
    def _load_qa(self, dataset: Dataset):
        def process_qa(example):
            return {
                "input": f"Context: {example['context']}\nQuestion: {example['question']}",
                "output": example['answers']['text'][0]
            }
        
        processed_dataset = dataset.map(process_qa)
        config = {
            "dataset_name": "squad",
            "task": task,
            "task_type": "extractive_qa",
            "metrics": ["exact_match", "f1"],
            "prompt_template": {
                "system": "Answer the question based on the given context.",
                "user": "{input}",
                "assistant": "Answer: {output}"
            }
        }
        return processed_dataset, config
    
    def _load_verification(self, dataset: Dataset):
        def process_verification(example):
            return {
                "input": f"Question: {example['question']}\nProposed Answer: {example['answers']['text'][0]}\nContext: {example['context']}",
                "output": "correct"  # 简化处理
            }
        
        processed_dataset = dataset.map(process_verification)
        config = {
            "dataset_name": "squad",
            "task": task,
            "task_type": "classification",
            "metrics": ["accuracy"],
            "prompt_template": {
                "system": "Verify if the proposed answer is correct based on the context.",
                "user": "{input}",
                "assistant": "The answer is {output}"
            }
        }
        return processed_dataset, config
    
    def _load_generation(self, dataset: Dataset):
        def process_generation(example):
            return {
                "input": example['context'],
                "output": example['question']
            }
        
        processed_dataset = dataset.map(process_generation)
        config = {
            "dataset_name": "squad",
            "task": task,
            "task_type": "text_generation",
            "metrics": ["bleu", "rouge"],
            "prompt_template": {
                "system": "Generate a question based on the given context.",
                "user": "Context: {input}",
                "assistant": "Question: {output}"
            }
        }
        return processed_dataset, config

    def load(self, task: str = "default") -> Tuple[Dataset, Dict[str, Any]]:
        if task not in self.get_supported_tasks():
            raise ValueError(f"Task {task} not supported. Available tasks: {self.get_supported_tasks()}")
        
        dataset = load_dataset("squad")
        
        if task == "qa":
            return self._load_qa(dataset)
        elif task == "verification":
            return self._load_verification(dataset)
        elif task == "generation":
            return self._load_generation(dataset)

class GPQALoader(BaseDataLoader):
    """单任务数据集"""
    def get_supported_tasks(self) -> List[str]:
        return ["default"]

    def load(self, task: str = "default") -> Tuple[Dataset, Dict[str, Any]]:
        if task != "default":
            raise ValueError("GPQA only supports default task")
        
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")
        
        def process_example(example):
            return {
                "input": example["Question"],
                "output": example["Correct Answer"]
            }
        
        processed_dataset = dataset.map(process_example)
        config = {
            "dataset_name": "gpqa",
            "task": "default",
            "task_type": "qa",
            "metrics": ["accuracy"],
            "prompt_template": {
                "system": "You are an expert in question answering. Answer the question with your best effort.",
                "user": "Question: {input}",
                "assistant": "Answer: {output}"
            }
        }
        return processed_dataset, config

class TRECLoader(BaseDataLoader):
    """单任务数据集"""
    def get_supported_tasks(self) -> List[str]:
        return ["default"]

    def load(self, task: str = "default") -> Tuple[Dataset, Dict[str, Any]]:
        if task != "default":
            raise ValueError("TREC only supports default task")
        
        dataset = load_dataset("trec")
        
        label_map = {
            0: "Abbreviation",
            1: "Entity",
            2: "Description",
            3: "Person",
            4: "Location",
            5: "Number"
        }
        
        def process_example(example):
            return {
                "input": example["text"],
                "output": label_map[example["coarse_label"]]
            }
        
        processed_dataset = dataset.map(process_example)
        config = {
            "dataset_name": "trec",
            "task": "default",
            "task_type": "classification",
            "metrics": ["accuracy"],
            "num_classes": 6,
            "prompt_template": {
                "system": "Classify the following question based on whether its answer type is a Number, Location, Person, Description, Entity, or Abbreviation.",
                "user": "Question: {input}",
                "assistant": "Type: {output}"
            }
        }
        return processed_dataset, config

class AGNewsLoader(BaseDataLoader):
    """单任务数据集"""
    def get_supported_tasks(self) -> List[str]:
        return ["default"]

    def load(self, task: str = "default") -> Tuple[Dataset, Dict[str, Any]]:
        if task != "default":
            raise ValueError("AGNews only supports default task")
        
        dataset = load_dataset("fancyzhx/ag_news")
        
        label_map = {
            0: "World",
            1: "Sports",
            2: "Sci/Tech",
            3: "Business"
        }
        
        def process_example(example):
            return {
                "input": example["text"],
                "output": label_map[example["label"]]
            }
        
        processed_dataset = dataset.map(process_example)
        config = {
            "dataset_name": "agnews",
            "task": "default",
            "task_type": "classification",
            "metrics": ["accuracy"],
            "num_classes": 4,
            "prompt_template": {
                "system": "Classify the news article into one of these categories: World, Sports, Sci/Tech, or Business.",
                "user": "Article: {input}",
                "assistant": "Category: {output}"
            }
        }
        return processed_dataset, config

class LexGlueLoader(BaseDataLoader):
    """用于加载和处理法律文本预测数据集的加载器"""
    
    def get_supported_tasks(self) -> List[str]:
        return ["prediction", "generation", "multiple_choice"]
    
    def _process_for_prediction(self, example):
        """处理预测任务的数据，将所有可能的endings作为选项"""
        # 生成选项列表
        choices = [f"Option {i}: {ending}" for i, ending in enumerate(example['endings'])]
        choices_text = "\n".join(choices)
        
        return {
            "input": f"Context: {example['context']}\n\nPossible Holdings:\n{choices_text}",
            "output": f"Option {example['label']}",
            "metadata": {
                "num_choices": len(example['endings']),
                "all_endings": example['endings'],
                "correct_index": example['label']
            }
        }
    
    def _process_for_multiple_choice(self, example):
        """处理多选题形式的数据"""
        return {
            "context": example['context'],
            "choices": example['endings'],
            "answer_index": example['label']
        }
    
    def _process_for_generation(self, example):
        """处理生成任务的数据"""
        return {
            "input": example['context'],
            "output": example['endings'][example['label']]  # 使用正确的ending作为目标输出
        }
    
    def load(self, task: str = "prediction") -> Tuple[Dataset, Dict[str, Any]]:
        """
        加载并处理法律文本数据集
        
        Args:
            task: 任务类型，支持：
                - "prediction": 预测正确的法律判决
                - "generation": 生成法律判决
                - "multiple_choice": 多选题形式的任务
        """
        if task not in self.get_supported_tasks():
            raise ValueError(f"Task {task} not supported. Available tasks: {self.get_supported_tasks()}")
        
        # 加载数据集
        dataset = load_dataset("coastalcph/lex_glue", "case_hold")
        
        if task == "prediction":
            processed_dataset = dataset.map(self._process_for_prediction)
            config = {
                "dataset_name": "lex_glue_case_hold",
                "task": task,
                "task_type": "multiple_choice_qa",
                "metrics": ["accuracy", "f1"],
                "prompt_template": {
                    "system": "You are a legal expert. Based on the given legal context, select the most appropriate legal holding from the provided options.",
                    "user": "{input}",
                    "assistant": "Based on the legal context, the correct holding is {output}."
                }
            }
        elif task == "multiple_choice":
            processed_dataset = dataset.map(self._process_for_multiple_choice)
            config = {
                "dataset_name": "lex_glue_case_hold",
                "task": task,
                "task_type": "multiple_choice",
                "metrics": ["accuracy"],
                "prompt_template": {
                    "system": "Select the correct legal holding for the given case.",
                    "user": "Context: {context}\nChoices: {choices}",
                    "assistant": "The correct choice is option {answer_index}."
                }
            }
        else:  # generation task
            processed_dataset = dataset.map(self._process_for_generation)
            config = {
                "dataset_name": "lex_glue_case_hold",
                "task": "generation",
                "task_type": "text_generation",
                "metrics": ["rouge", "bleu"],
                "prompt_template": {
                    "system": "You are a legal expert. Generate an appropriate legal holding for the given case.",
                    "user": "Legal Context: {input}",
                    "assistant": "Legal Holding: {output}"
                }
            }
        
        return processed_dataset, config

class MedNLILoader(BaseDataLoader):
    """用于加载和处理医疗自然语言推理数据集的加载器，该数据集必须先手动下载"""
    
    def get_supported_tasks(self) -> List[str]:
        return ["nli", "classification", "explanation"]
    
    def _download_and_prepare_data(self, data_dir: str) -> Dataset:
        """
        准备数据集
        Args:
            data_dir: 包含mli_train.jsonl等文件的目录路径
        """
        # 检查必要的文件是否存在
        required_files = ['mli_train.jsonl', 'mli_dev.jsonl', 'mli_test.jsonl']
        for file in required_files:
            if not os.path.exists(os.path.join(data_dir, file)):
                raise FileNotFoundError(f"Required file {file} not found in {data_dir}")
        
        # 加载数据集
        dataset = load_dataset('json',
                             data_files={
                                 'train': os.path.join(data_dir, 'mli_train.jsonl'),
                                 'validation': os.path.join(data_dir, 'mli_dev.jsonl'),
                                 'test': os.path.join(data_dir, 'mli_test.jsonl')
                             })
        
        return dataset
    
    def _process_for_nli(self, example):
        """处理NLI任务的数据"""
        return {
            "input": f"Premise: {example['sentence1']}\nHypothesis: {example['sentence2']}",
            "output": example['gold_label'],
            "metadata": {
                "pairID": example.get('pairID', ''),
                "premise": example['sentence1'],
                "hypothesis": example['sentence2']
            }
        }
    
    def _process_for_classification(self, example):
        """处理分类任务的数据"""
        label_map = {
            'entailment': 'The hypothesis logically follows from the premise',
            'contradiction': 'The hypothesis contradicts the premise',
            'neutral': 'The hypothesis is neither supported nor contradicted by the premise'
        }
        
        return {
            "input": f"Medical Context: {example['sentence1']}\nStatement: {example['sentence2']}",
            "output": label_map[example['gold_label']],
            "label": example['gold_label']
        }
    
    def load(self, task: str = "nli", data_dir: str = "./mednli_data") -> Tuple[Dataset, Dict[str, Any]]:
        """
        加载并处理MedNLI数据集
        
        Args:
            task: 任务类型，支持 "nli"/"classification"/"explanation"
            data_dir: 数据集文件所在目录
            
        Returns:
            处理后的数据集和配置信息
        """
        if task not in self.get_supported_tasks():
            raise ValueError(f"Task {task} not supported. Available tasks: {self.get_supported_tasks()}")
        
        # 加载原始数据集
        dataset = self._download_and_prepare_data(data_dir)
        
        # 根据任务类型选择处理方法
        if task == "nli":
            processed_dataset = dataset.map(self._process_for_nli)
            config = {
                "dataset_name": "mednli",
                "task": task,
                "task_type": "natural_language_inference",
                "metrics": ["accuracy", "f1"],
                "labels": ["entailment", "contradiction", "neutral"],
                "prompt_template": {
                    "system": "You are a medical expert. Determine the logical relationship between the premise and hypothesis.",
                    "user": "{input}",
                    "assistant": "The relationship is: {output}"
                }
            }
        else:  # classification task
            processed_dataset = dataset.map(self._process_for_classification)
            config = {
                "dataset_name": "mednli",
                "task": task,
                "task_type": "classification",
                "metrics": ["accuracy"],
                "num_classes": 3,
                "prompt_template": {
                    "system": "Analyze the logical relationship between the medical context and the statement.",
                    "user": "{input}",
                    "assistant": "{output}"
                }
            }
        
        return processed_dataset, config

class PubMedQALoader(BaseDataLoader):
    """用于加载和处理医学文献问答数据集的加载器"""
    
    def get_supported_tasks(self) -> List[str]:
        return [
            "qa",              # 问答任务
            "classification",  # 文本分类（段落标签预测）
            "summarization",   # 总结生成
            "mesh_prediction"  # 医学主题词预测
        ]
    
    def _process_for_qa(self, example):
        """处理问答任务的数据"""
        # 将上下文段落组合，并保持原有的结构标签
        structured_context = []
        for ctx, label in zip(example['CONTEXTS'], example['LABELS']):
            structured_context.append(f"{label}: {ctx}")
        
        return {
            "input": f"Question: {example['QUESTION']}\n\nContext:\n{chr(10).join(structured_context)}",
            "output": example['LONG_ANSWER'],
            "metadata": {
                "meshes": example['MESHES'],
                "year": example['YEAR'],
                "context_labels": example['LABELS']
            }
        }
    
    def _process_for_classification(self, example):
        """处理段落分类任务的数据"""
        # 为每个段落创建单独的样本
        processed_examples = []
        for ctx, label in zip(example['CONTEXTS'], example['LABELS']):
            processed_examples.append({
                "input": ctx,
                "output": label,
                "metadata": {
                    "question": example['QUESTION'],
                    "meshes": example['MESHES']
                }
            })
        return processed_examples
    
    def _process_for_summarization(self, example):
        """处理摘要生成任务的数据"""
        return {
            "input": chr(10).join(example['CONTEXTS']),
            "output": example['LONG_ANSWER'],
            "metadata": {
                "question": example['QUESTION'],
                "meshes": example['MESHES'],
                "labels": example['LABELS']
            }
        }
    
    def _process_for_mesh_prediction(self, example):
        """处理医学主题词预测任务的数据"""
        return {
            "input": f"Question: {example['QUESTION']}\n\nContext:\n{chr(10).join(example['CONTEXTS'])}",
            "output": ", ".join(example['MESHES']),
            "metadata": {
                "original_meshes": example['MESHES'],
                "answer": example['LONG_ANSWER']
            }
        }
    
    def load(self, task: str = "qa") -> Tuple[Dataset, Dict[str, Any]]:
        """
        加载并处理医学文献问答数据集
        
        Args:
            task: 任务类型，支持 "qa"/"classification"/"summarization"/"mesh_prediction"
            
        Returns:
            处理后的数据集和配置信息
        """
        if task not in self.get_supported_tasks():
            raise ValueError(f"Task {task} not supported. Available tasks: {self.get_supported_tasks()}")
        
        # 根据任务类型选择处理方法和配置
        processing_fn = getattr(self, f"_process_for_{task}")
        
        task_configs = {
            "qa": {
                "task_type": "question_answering",
                "metrics": ["rouge", "bleu", "meteor"],
                "prompt_template": {
                    "system": "You are a medical expert. Answer the question based on the provided context.",
                    "user": "{input}",
                    "assistant": "Based on the provided information: {output}"
                }
            },
            "classification": {
                "task_type": "text_classification",
                "metrics": ["accuracy", "f1"],
                "labels": ["BACKGROUND", "METHODS", "RESULTS"],
                "prompt_template": {
                    "system": "Classify the given medical text passage into its appropriate section type.",
                    "user": "Text: {input}",
                    "assistant": "Section: {output}"
                }
            },
            "summarization": {
                "task_type": "text_summarization",
                "metrics": ["rouge", "bleu"],
                "prompt_template": {
                    "system": "Summarize the key findings from the medical research text.",
                    "user": "{input}",
                    "assistant": "Summary: {output}"
                }
            },
            "mesh_prediction": {
                "task_type": "multi_label_classification",
                "metrics": ["precision", "recall", "f1"],
                "prompt_template": {
                    "system": "Predict relevant Medical Subject Headings (MeSH terms) for the given medical text.",
                    "user": "{input}",
                    "assistant": "Relevant MeSH terms: {output}"
                }
            }
        }

        dataset = load_dataset('bigbio/pubmed_qa')  # 需要替换为实际的数据集路径
        processed_dataset = dataset.map(processing_fn)
        
        config = {
            "dataset_name": "medical_qa",
            "task": task,
            **task_configs[task],
            "additional_features": {
                "has_mesh_terms": True,
                "has_structured_context": True,
                "has_long_answer": True
            }
        }
        
        return processed_dataset, config

# 数据集加载器注册表
DATASET_LOADERS = {
    "squad": SQuADLoader,
    "gpqa": GPQALoader,
    "trec": TRECLoader,
    "agnews": AGNewsLoader,
    "lexglue": LexGlueLoader,
    "pubmedqa": PubMedQALoader
}

def get_available_tasks(dataset_name: str) -> List[str]:
    """获取指定数据集支持的所有任务"""
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {list(DATASET_LOADERS.keys())}")
    
    loader: BaseDataLoader = DATASET_LOADERS[dataset_name]()
    return loader.get_supported_tasks()

def load_dataset_and_config(dataset_name: str, task: str = "default") -> Tuple[Dataset, Dict[str, Any]]:
    """
    统一的数据集加载接口
    
    Args:
        dataset_name: 数据集名称
        task: 任务名称，默认为"default"
    
    Returns:
        处理后的数据集和配置信息的元组
    
    Raises:
        ValueError: 如果数据集名称未注册或任务不支持
    """
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {list(DATASET_LOADERS.keys())}")
    
    loader: BaseDataLoader = DATASET_LOADERS[dataset_name]()
    supported_tasks = loader.get_supported_tasks()
    
    if task not in supported_tasks:
        raise ValueError(f"Task {task} not supported for dataset {dataset_name}. Available tasks: {supported_tasks}")
    
    return loader.load(task)

# 使用示例
if __name__ == "__main__":
    # 测试多任务数据集
    print("SQuAD supported tasks:", get_available_tasks("squad"))
    for task in get_available_tasks("squad"):
        dataset, config = load_dataset_and_config("squad", task)
        print(f"\nSQuAD {task} task config:", config)
        print(f"SQuAD {task} example:", dataset['train'][0])
    
    # 测试单任务数据集
    single_task_datasets = ["gpqa", "trec", "agnews"]
    for dataset_name in single_task_datasets:
        print(f"\n{dataset_name} supported tasks:", get_available_tasks(dataset_name))
        dataset, config = load_dataset_and_config(dataset_name)
        print(f"{dataset_name} Config:", config)
        print(f"{dataset_name} Example:", dataset['train'][0])