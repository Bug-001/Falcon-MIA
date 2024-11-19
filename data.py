from typing import Dict, Tuple, Any, List
from datasets import load_dataset, Dataset, DatasetDict
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from pathlib import Path
import requests
from tqdm import tqdm
import xml.etree.ElementTree as ET
import os

class BaseDataLoader(ABC):
    """数据集加载器的基类"""
    @abstractmethod
    def load(self, task: str = "default", **kwargs) -> Tuple[DatasetDict, Dict[str, Any]]:
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

class GPQALoader(BaseDataLoader):
    """单任务数据集"""
    def get_supported_tasks(self) -> List[str]:
        return ["default"]

    def load(self, task: str = "default", **kwargs) -> Tuple[DatasetDict, Dict[str, Any]]:
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

    def load(self, task: str = "default", **kwargs) -> Tuple[DatasetDict, Dict[str, Any]]:
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

    def load(self, task: str = "default", **kwargs) -> Tuple[DatasetDict, Dict[str, Any]]:
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
    
    def load(self, task: str = "prediction", **kwargs) -> Tuple[DatasetDict, Dict[str, Any]]:
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
    
    def _prepare_data(self, data_dir: str) -> Dataset:
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
    
    def load(self, task: str = "nli", **kwargs) -> Tuple[DatasetDict, Dict[str, Any]]:
        """
        加载并处理MedNLI数据集
        
        Args:
            task: 任务类型，支持 "nli"/"classification"/"explanation"
            
        Returns:
            处理后的数据集和配置信息
        """
        if task not in self.get_supported_tasks():
            raise ValueError(f"Task {task} not supported. Available tasks: {self.get_supported_tasks()}")
        
        # 加载原始数据集
        data_dir = os.path.join(kwargs.get("data_dir"), 'mednli')
        dataset = self._prepare_data(data_dir)
        
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
    
    def _process_for_classification(self, batch):
        """处理段落分类任务的数据"""
        # 为每个段落创建单独的样本
        result = {
            "input": [],
            "output": [],
            "metadata": []
        }

        for i in range(len(batch['CONTEXTS'])):
            for ctx, label in zip(batch['CONTEXTS'][i], batch['LABELS'][i]):
                result["input"].append(ctx)
                result["output"].append(label)
                result["metadata"].append({
                    "question": batch['QUESTION'][i],
                    "meshes": batch['MESHES'][i]
                })
    
        return result
    
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
    
    def load(self, task: str = "qa", **kwargs) -> Tuple[DatasetDict, Dict[str, Any]]:
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

        datadict = load_dataset('bigbio/pubmed_qa')  # 需要替换为实际的数据集路径
        # 简洁的字典推导式方式
        batched = False
        if processing_fn == self._process_for_classification:
            batched = True
        processed_dataset = DatasetDict({
            split_name: dataset.map(
                processing_fn,
                batched=batched,
                remove_columns=dataset.column_names,
            )
            for split_name, dataset in datadict.items()
        })
        
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

class CCELoader(BaseDataLoader):
    """用于加载和处理CCE(Common Configuration Enumeration)安全配置数据集的加载器"""
    
    def get_supported_tasks(self) -> List[str]:
        return [
            "classification",  # 配置类型分类
            "requirement_extraction",  # 提取配置要求
            "reference_prediction",  # 预测相关安全标准引用
            "platform_detection"  # 检测适用平台
        ]
    
    def _parse_xml_data(self, xml_file: str) -> List[Dict]:
        """解析CCE XML文件并提取相关信息"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 处理命名空间
        namespaces = {
            'cce': 'http://cce.mitre.org',
            'dc': 'http://purl.org/dc/terms/'
        }
        
        entries = []
        for cce in root.findall('.//cce:cce', namespaces):
            entry = {
                'cce_id': cce.get('cce_id', ''),
                'platform': cce.get('platform', ''),
                'modified': cce.get('modified', ''),
                'description': cce.find('.//cce:description', namespaces).text if cce.find('.//cce:description', namespaces) is not None else '',
                'technical_mechanisms': [mech.text for mech in cce.findall('.//cce:technical_mechanism', namespaces) if mech.text is not None],
                'parameters': [param.text for param in cce.findall('.//cce:parameter', namespaces)],
                'references': [{'id': ref.get('resource_id', ''), 'text': ref.text} for ref in cce.findall('.//cce:reference', namespaces)]
            }
            entries.append(entry)
        
        return entries
    
    def _process_for_classification(self, example):
        """处理配置类型分类任务的数据"""
        # 基于technical_mechanisms和description判断配置类型
        config_types = {
            'filesystem': ['fstab', 'filesystem', 'mount'],
            'registry': ['registry', 'HKEY_LOCAL_MACHINE', 'regedit'],
            'permissions': ['permissions', 'dacl', 'acl'],
            'network': ['network', 'firewall', 'tcp'],
            'authentication': ['password', 'login', 'auth']
        }
        
        text = f"{example['description']} {' '.join(example['technical_mechanisms'])}"
        config_type = 'other'
        for type_name, keywords in config_types.items():
            if any(keyword.lower() in text.lower() for keyword in keywords):
                config_type = type_name
                break
        
        return {
            "input": example['description'],
            "output": config_type,
            "metadata": {
                "cce_id": example['cce_id'],
                "platform": example['platform']
            }
        }
    
    def _process_for_requirement_extraction(self, example):
        """处理配置要求提取任务的数据"""
        return {
            "input": example['description'],
            "output": {
                "technical_mechanisms": example['technical_mechanisms'],
                "parameters": example['parameters']
            },
            "metadata": {
                "cce_id": example['cce_id'],
                "platform": example['platform']
            }
        }
    
    def _process_for_reference_prediction(self, example):
        """处理安全标准引用预测任务的数据"""
        references = [ref['text'] for ref in example['references']]
        return {
            "input": f"{example['description']}\n{' '.join(example['technical_mechanisms'])}",
            "output": references,
            "metadata": {
                "cce_id": example['cce_id'],
                "reference_ids": [ref['id'] for ref in example['references']]
            }
        }
    
    def _process_for_platform_detection(self, example):
        """处理平台检测任务的数据"""
        return {
            "input": f"{example['description']}\n{' '.join(example['technical_mechanisms'])}",
            "output": example['platform'],
            "metadata": {
                "cce_id": example['cce_id'],
                "modified_date": example['modified']
            }
        }
    
    def load(self, task: str = "classification", **kwargs) -> Tuple[DatasetDict, Dict[str, Any]]:
        """
        加载并处理CCE数据集
        
        Args:
            task: 任务类型，支持 "classification"/"requirement_extraction"/"reference_prediction"/"platform_detection"
            data_path: CCE XML数据文件路径
            
        Returns:
            处理后的数据集和配置信息
        """
        if task not in self.get_supported_tasks():
            raise ValueError(f"Task {task} not supported. Available tasks: {self.get_supported_tasks()}")
        
        # 解析XML数据
        data_name: str = "cce-COMBINED-5.20130214.xml"
        cce_dir = os.path.join(kwargs.get("data_dir"), 'cce')
        data_path = os.path.join(cce_dir, data_name)
        try:
            raw_data = self._parse_xml_data(data_path)
        except FileNotFoundError:
            os.makedirs(cce_dir, exist_ok=True)
            # 发送GET请求，流式传输
            url = "https://cce.mitre.org/lists/data/downloads/cce-COMBINED-5.20130214.xml"
            response = requests.get(url, stream=True)
            response.raise_for_status()  # 确保请求成功
            
            # 获取文件大小（如果服务器提供）
            total_size = int(response.headers.get('content-length', 0))
            
            # 打开文件并写入数据
            with open(data_path, 'wb') as file, tqdm(
                desc=f'Downloading {data_name}',
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    progress_bar.update(size)
            raw_data = self._parse_xml_data(data_path)
        
        # 根据任务类型选择处理方法
        processing_fn = getattr(self, f"_process_for_{task}")
        
        # 创建数据集
        processed_data = [processing_fn(entry) for entry in raw_data]
        dataset = DatasetDict({
            "train": Dataset.from_list(processed_data)
        })
        
        # 任务配置
        task_configs = {
            "classification": {
                "task_type": "text_classification",
                "metrics": ["accuracy", "f1"],
                "prompt_template": {
                    "system": "Classify the security configuration type based on the description.",
                    "user": "Configuration: {input}",
                    "assistant": "Configuration type: {output}"
                }
            },
            "requirement_extraction": {
                "task_type": "information_extraction",
                "metrics": ["precision", "recall", "f1"],
                "prompt_template": {
                    "system": "Extract the technical mechanisms and parameters from the security configuration.",
                    "user": "{input}",
                    "assistant": "Technical mechanisms: {output[technical_mechanisms]}\nParameters: {output[parameters]}"
                }
            },
            "reference_prediction": {
                "task_type": "multi_label_classification",
                "metrics": ["precision", "recall", "f1"],
                "prompt_template": {
                    "system": "Predict relevant security standards and guidelines for the configuration.",
                    "user": "{input}",
                    "assistant": "Relevant standards: {output}"
                }
            },
            "platform_detection": {
                "task_type": "single_label_classification",
                "metrics": ["accuracy"],
                "prompt_template": {
                    "system": "Determine the platform this security configuration applies to.",
                    "user": "{input}",
                    "assistant": "Platform: {output}"
                }
            }
        }
        
        config = {
            "dataset_name": "cce_security",
            "task": task,
            **task_configs[task],
            "additional_features": {
                "has_cce_id": True,
                "has_technical_mechanisms": True,
                "has_references": True
            }
        }
        
        return dataset, config

# 数据集加载器注册表
DATASET_LOADERS = {
    "gpqa": GPQALoader,
    "trec": TRECLoader,
    "agnews": AGNewsLoader,
    "lexglue": LexGlueLoader,
    "pubmedqa": PubMedQALoader,
    "cce": CCELoader,
}