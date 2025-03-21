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
    """Base class for dataset loaders"""
    @abstractmethod
    def load(self, task: str = "default", **kwargs) -> Tuple[DatasetDict, Dict[str, Any]]:
        """
        Load and process dataset
        Args:
            task: Task name, default is "default" representing a single-task dataset
        Returns:
            Tuple of processed dataset and configuration
        """
        pass

    @abstractmethod
    def get_supported_tasks(self) -> List[str]:
        """Return a list of all supported tasks for this dataset"""
        pass

class GPQALoader(BaseDataLoader):
    """Single-task dataset"""
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
            "prompt_template": [
                {
                    "system": "You are an expert in question answering. Answer the question with your best effort.",
                    "user": "Question: {input}",
                    "assistant": "Answer: {output}"
                },
                {
                    "system": "As a knowledgeable assistant, provide accurate and concise answers to questions.",
                    "user": "Please answer this question: {input}",
                    "assistant": "The answer is: {output}"
                },
                {
                    "system": "You are a helpful AI trained to provide precise answers to various questions.",
                    "user": "Here's a question for you: {input}",
                    "assistant": "Here's the answer: {output}"
                },
                {
                    "system": "Acting as a question-answering specialist, give detailed and accurate responses.",
                    "user": "I need an answer to this: {input}",
                    "assistant": "Based on my knowledge: {output}"
                }
            ]
        }
        return processed_dataset, config

class TRECLoader(BaseDataLoader):
    """Single-task dataset"""
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
            "label_map": label_map,
            "num_classes": 6,
            "prompt_template": [
                {
                    "system": "Classify the following question based on whether its answer type is a Number, Location, Person, Description, Entity, or Abbreviation.",
                    "user": "Question: {input}",
                    "assistant": "Type: {output}"
                },
                {
                    "system": "Determine the expected answer type for the given question from these categories: Number, Location, Person, Description, Entity, or Abbreviation.",
                    "user": "Analyze this question: {input}",
                    "assistant": "The expected answer type is: {output}"
                },
                {
                    "system": "You are an expert at analyzing questions. Categorize each question based on its expected answer type.",
                    "user": "What type of answer does this question seek: {input}",
                    "assistant": "This question seeks a(n) {output} as its answer."
                }
            ]
        }
        return processed_dataset, config

class AGNewsLoader(BaseDataLoader):
    """Single-task dataset"""
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
            "label_map": label_map,
            "prompt_template": [
                {
                    "system": "Classify the news article into one of these categories: World, Sports, Sci/Tech, or Business.",
                    "user": "Article: {input}",
                    "assistant": "Category: {output}"
                },
                {
                    "system": "As a news categorization expert, classify each article into its appropriate category.",
                    "user": "Please categorize this news article:\n{input}",
                    "assistant": "This article belongs to the {output} category."
                },
                {
                    "system": "You are a news classifier. Determine the most suitable category for each article from: World, Sports, Sci/Tech, or Business.",
                    "user": "News text: {input}",
                    "assistant": "Based on the content, this is a {output} news article."
                }
            ]
        }
        return processed_dataset, config

class LexGlueLoader(BaseDataLoader):
    """Loader for loading and processing legal text prediction datasets"""
    
    def get_supported_tasks(self) -> List[str]:
        return ["prediction", "generation", "multiple_choice", "judgment"]
    
    def _process_for_prediction(self, example):
        """Process data for prediction tasks, using all possible endings as options"""
        # Generate options list
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
        """Process data in multiple choice format"""
        return {
            "context": example['context'],
            "choices": example['endings'],
            "answer_index": example['label']
        }
    
    def _process_for_judgment(self, example):
        """Process data for judgment tasks"""
        context = example['context']
        selected_choice = abs(hash(context)) // 10 % len(example['endings'])
        return {
            "input": context + ' ' + example['endings'][selected_choice],
            "output": 'Yes' if selected_choice == example['label'] else 'No'
        }
    
    def _process_for_generation(self, example):
        """Process data for generation tasks"""
        return {
            "input": example['context'],
            "output": example['endings'][example['label']]  # Use the correct ending as the target output
        }
    
    def load(self, task: str = "prediction", **kwargs) -> Tuple[DatasetDict, Dict[str, Any]]:
        """
        Load and process legal text datasets
        
        Args:
            task: Task type, supports:
                - "prediction": Predict the correct legal holding
                - "generation": Generate legal holdings
                - "multiple_choice": Task in multiple choice format
        """
        if task not in self.get_supported_tasks():
            raise ValueError(f"Task {task} not supported. Available tasks: {self.get_supported_tasks()}")
        
        # Load dataset
        dataset = load_dataset("coastalcph/lex_glue", "case_hold")
        
        if task == "prediction":
            processed_dataset = dataset.map(self._process_for_prediction)
            config = {
                "dataset_name": "lex_glue_case_hold",
                "task": task,
                "task_type": "multiple_choice_qa",
                "metrics": ["accuracy", "f1"],
                "prompt_template": [
                    {
                        "system": "You are a legal expert. Based on the given legal context, select the most appropriate legal holding from the provided options.",
                        "user": "{input}",
                        "assistant": "Based on the legal context, the correct holding is {output}."
                    },
                    {
                        "system": "As an experienced legal analyst, evaluate the case and select the most fitting legal holding.",
                        "user": "Case details:\n{input}",
                        "assistant": "After analyzing the case, I determine that {output} is the correct legal holding."
                    },
                    {
                        "system": "You are a legal professional specializing in case analysis. Choose the most appropriate holding for the given case.",
                        "user": "Review this case:\n{input}",
                        "assistant": "Having reviewed the case, I conclude that {output} represents the correct holding."
                    }
                ]
            }
        elif task == "multiple_choice":
            processed_dataset = dataset.map(self._process_for_multiple_choice)
            config = {
                "dataset_name": "lex_glue_case_hold",
                "task": task,
                "task_type": "multiple_choice",
                "metrics": ["accuracy"],
                "prompt_template": [
                    {
                        "system": "Select the correct legal holding for the given case.",
                        "user": "Context: {context}\nChoices: {choices}",
                        "assistant": "The correct choice is option {answer_index}."
                    },
                    {
                        "system": "As a legal expert, identify the most appropriate legal holding from the given options.",
                        "user": "Case context:\n{context}\nAvailable options:\n{choices}",
                        "assistant": "After careful consideration, I select option {answer_index} as the correct holding."
                    },
                    {
                        "system": "You are a legal professional. Choose the correct legal holding from multiple options.",
                        "user": "Legal case:\n{context}\nPossible holdings:\n{choices}",
                        "assistant": "Based on the case details, option {answer_index} is the correct legal holding."
                    }
                ]
            }
        elif task == "judgment":
            processed_dataset = dataset.map(self._process_for_judgment)
            config = {
                "dataset_name": "lex_glue_case_hold",
                "task": task,
                "task_type": "binary_classification",
                "metrics": ["accuracy"],
                "prompt_template": [
                    {
                        "system": "You are a legal expert. Determine if the provided legal holding is correct for the given case, and answer with Yes or No.",
                        "user": "{input}",
                        "assistant": "{output}"
                    },
                    {
                        "system": "As a legal analyst, evaluate whether the given legal holding correctly applies to this case.",
                        "user": "Case and holding:\n{input}",
                        "assistant": "Is this holding correct? {output}"
                    },
                    {
                        "system": "You are a legal professional. Assess if the proposed holding matches the case facts.",
                        "user": "Review this case and holding:\n{input}",
                        "assistant": "The holding is {output} correct for this case."
                    }
                ]
            }
        else:  # generation task
            processed_dataset = dataset.map(self._process_for_generation)
            config = {
                "dataset_name": "lex_glue_case_hold",
                "task": "generation",
                "task_type": "text_generation",
                "metrics": ["rouge", "bleu"],
                "prompt_template": [
                    {
                        "system": "You are a legal expert. Generate an appropriate legal holding for the given case.",
                        "user": "Legal Context: {input}",
                        "assistant": "Legal Holding: {output}"
                    },
                    {
                        "system": "As an experienced legal professional, write a suitable legal holding based on the case details.",
                        "user": "Case details:\n{input}",
                        "assistant": "Based on the case, the appropriate holding is: {output}"
                    },
                    {
                        "system": "You are a legal writer specializing in drafting holdings. Create a holding that matches the case context.",
                        "user": "Review this case context:\n{input}",
                        "assistant": "After reviewing the case, I draft the following holding: {output}"
                    }
                ]
            }
        
        return processed_dataset, config

class MedNLILoader(BaseDataLoader):
    """Loader for loading and processing medical natural language inference datasets, which must be manually downloaded first"""
    
    def get_supported_tasks(self) -> List[str]:
        return ["nli", "classification", "explanation"]
    
    def _prepare_data(self, data_dir: str) -> Dataset:
        # Check if necessary files exist
        required_files = ['mli_train.jsonl', 'mli_dev.jsonl', 'mli_test.jsonl']
        for file in required_files:
            if not os.path.exists(os.path.join(data_dir, file)):
                raise FileNotFoundError(f"Required file {file} not found in {data_dir}")
        
        # Load dataset
        dataset = load_dataset('json',
                             data_files={
                                 'train': os.path.join(data_dir, 'mli_train.jsonl'),
                                 'validation': os.path.join(data_dir, 'mli_dev.jsonl'),
                                 'test': os.path.join(data_dir, 'mli_test.jsonl')
                             })
        
        return dataset
    
    def _process_for_nli(self, example):
        """Process data for NLI tasks"""
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
        """Process data for classification tasks"""
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
        Load and process MedNLI dataset
        
        Args:
            task: Task type, supports "nli"/"classification"/"explanation"
            
        Returns:
            Processed dataset and configuration information
        """
        if task not in self.get_supported_tasks():
            raise ValueError(f"Task {task} not supported. Available tasks: {self.get_supported_tasks()}")
        
        # Load original dataset
        data_dir = os.path.join(kwargs.get("data_dir"), 'mednli')
        dataset = self._prepare_data(data_dir)
        
        # Choose processing method based on task type
        if task == "nli":
            processed_dataset = dataset.map(self._process_for_nli)
            config = {
                "dataset_name": "mednli",
                "task": task,
                "task_type": "natural_language_inference",
                "metrics": ["accuracy", "f1"],
                "labels": ["entailment", "contradiction", "neutral"],
                "prompt_template": [
                    {
                        "system": "You are a medical expert. Determine the logical relationship between the premise and hypothesis.",
                        "user": "{input}",
                        "assistant": "The relationship is: {output}"
                    },
                    {
                        "system": "As a medical professional, analyze whether the hypothesis follows from, contradicts, or is neutral to the premise.",
                        "user": "Medical statements:\n{input}",
                        "assistant": "After analysis, the relationship between these statements is: {output}"
                    },
                    {
                        "system": "You are a clinical reasoning expert. Evaluate the logical connection between these medical statements.",
                        "user": "Please analyze:\n{input}",
                        "assistant": "Based on medical knowledge, these statements have a {output} relationship."
                    }
                ]
            }
        else:  # classification task
            processed_dataset = dataset.map(self._process_for_classification)
            config = {
                "dataset_name": "mednli",
                "task": task,
                "task_type": "classification",
                "metrics": ["accuracy"],
                "num_classes": 3,
                "prompt_template": [
                    {
                        "system": "Analyze the logical relationship between the medical context and the statement.",
                        "user": "{input}",
                        "assistant": "{output}"
                    },
                    {
                        "system": "As a medical expert, determine how the given statement relates to the medical context.",
                        "user": "Review these medical statements:\n{input}",
                        "assistant": "My analysis shows that {output}"
                    },
                    {
                        "system": "You are a healthcare professional. Evaluate the relationship between these medical statements.",
                        "user": "Medical context and statement:\n{input}",
                        "assistant": "From a clinical perspective, {output}"
                    }
                ]
            }
        
        return processed_dataset, config

class PubMedQALoader(BaseDataLoader):
    """Loader for loading and processing medical literature question answering datasets"""
    
    def get_supported_tasks(self) -> List[str]:
        return [
            "qa",              # Question answering task
            "classification",  # Text classification (paragraph label prediction)
            "summarization",   # Summary generation
            "mesh_prediction"  # Medical subject heading prediction
        ]
    
    def _process_for_qa(self, example):
        """Process data for question answering tasks"""
        # Combine context paragraphs while maintaining structure labels
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
        """Process data for paragraph classification tasks"""
        # Create separate samples for each paragraph
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
        """Process data for summary generation tasks"""
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
        """Process data for medical subject heading prediction tasks"""
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
        Load and process medical literature question answering dataset
        
        Args:
            task: Task type, supports "qa"/"classification"/"summarization"/"mesh_prediction"
            
        Returns:
            Processed dataset and configuration information
        """
        if task not in self.get_supported_tasks():
            raise ValueError(f"Task {task} not supported. Available tasks: {self.get_supported_tasks()}")
        
        # Select processing method and configuration based on task type
        processing_fn = getattr(self, f"_process_for_{task}")
        
        task_configs = {
            "qa": {
                "task_type": "question_answering",
                "metrics": ["rouge", "bleu", "meteor"],
                "prompt_template": [
                    {
                        "system": "You are a medical expert. Answer the question based on the provided context.",
                        "user": "{input}",
                        "assistant": "Based on the provided information: {output}"
                    },
                    {
                        "system": "As a medical researcher, provide an evidence-based answer using the given context.",
                        "user": "Research question:\n{input}",
                        "assistant": "According to the medical literature: {output}"
                    },
                    {
                        "system": "You are a healthcare professional specializing in medical literature analysis.",
                        "user": "Please answer based on this context:\n{input}",
                        "assistant": "After analyzing the medical literature, I conclude: {output}"
                    }
                ]
            },
            "classification": {
                "task_type": "text_classification",
                "metrics": ["accuracy", "f1"],
                "labels": ["BACKGROUND", "METHODS", "RESULTS"],
                "prompt_template": [
                    {
                        "system": "Classify the given medical text passage into its appropriate section type.",
                        "user": "Text: {input}",
                        "assistant": "Section: {output}"
                    },
                    {
                        "system": "As a medical paper expert, identify which section this text belongs to.",
                        "user": "Medical text passage:\n{input}",
                        "assistant": "This passage belongs to the {output} section."
                    },
                    {
                        "system": "You are a medical research paper analyst. Categorize this text into the correct section.",
                        "user": "Analyze this passage:\n{input}",
                        "assistant": "This text is part of the {output} section of the research paper."
                    }
                ]
            },
            "summarization": {
                "task_type": "text_summarization",
                "metrics": ["rouge", "bleu"],
                "prompt_template": [
                    {
                        "system": "Summarize the key findings from the medical research text.",
                        "user": "{input}",
                        "assistant": "Summary: {output}"
                    },
                    {
                        "system": "As a medical researcher, create a concise summary of the main points.",
                        "user": "Medical text to summarize:\n{input}",
                        "assistant": "Key findings: {output}"
                    },
                    {
                        "system": "You are a medical literature expert. Provide a comprehensive yet concise summary.",
                        "user": "Please summarize this medical text:\n{input}",
                        "assistant": "Research summary: {output}"
                    }
                ]
            },
            "mesh_prediction": {
                "task_type": "multi_label_classification",
                "metrics": ["precision", "recall", "f1"],
                "prompt_template": [
                    {
                        "system": "Predict relevant Medical Subject Headings (MeSH terms) for the given medical text.",
                        "user": "{input}",
                        "assistant": "Relevant MeSH terms: {output}"
                    },
                    {
                        "system": "As a medical indexing expert, identify appropriate MeSH terms for this content.",
                        "user": "Medical content:\n{input}",
                        "assistant": "Applicable MeSH terms: {output}"
                    },
                    {
                        "system": "You are a MeSH terminology specialist. List relevant medical subject headings.",
                        "user": "Analyze this medical text:\n{input}",
                        "assistant": "Based on the content, the relevant MeSH terms are: {output}"
                    }
                ]
            }
        }

        datadict = load_dataset('bigbio/pubmed_qa')  # Need to replace with actual dataset path
        # Concise dictionary comprehension method
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
    """Loader for loading and processing CCE (Common Configuration Enumeration) security configuration datasets"""
    
    def get_supported_tasks(self) -> List[str]:
        return [
            "classification",  # Configuration type classification
            "requirement_extraction",  # Extract configuration requirements
            "reference_prediction",  # Predict relevant security standards and guidelines
            "platform_detection"  # Detect applicable platforms
        ]
    
    def _parse_xml_data(self, xml_file: str) -> List[Dict]:
        """Parse CCE XML file and extract relevant information"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Handle namespace
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
        """Process data for configuration type classification tasks"""
        # Based on technical_mechanisms and description, determine configuration type
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
        """Process data for configuration requirements extraction tasks"""
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
        """Process data for security standards and guidelines prediction tasks"""
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
        """Process data for platform detection tasks"""
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
        Load and process CCE data set
        
        Args:
            task: Task type, supports "classification"/"requirement_extraction"/"reference_prediction"/"platform_detection"
            data_path: CCE XML data file path
            
        Returns:
            Processed dataset and configuration information
        """
        if task not in self.get_supported_tasks():
            raise ValueError(f"Task {task} not supported. Available tasks: {self.get_supported_tasks()}")
        
        # Parse XML data
        data_name: str = "cce-COMBINED-5.20130214.xml"
        cce_dir = os.path.join(kwargs.get("data_dir"), 'cce')
        data_path = os.path.join(cce_dir, data_name)
        try:
            raw_data = self._parse_xml_data(data_path)
        except FileNotFoundError:
            os.makedirs(cce_dir, exist_ok=True)
            # Send GET request, stream transfer
            url = "https://cce.mitre.org/lists/data/downloads/cce-COMBINED-5.20130214.xml"
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Ensure request successful
            
            # Get file size (if server provides)
            total_size = int(response.headers.get('content-length', 0))
            
            # Open file and write data
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
        
        # Choose processing method based on task type
        processing_fn = getattr(self, f"_process_for_{task}")
        
        # Create dataset
        processed_data = [processing_fn(entry) for entry in raw_data]
        dataset = DatasetDict({
            "train": Dataset.from_list(processed_data)
        })
        
        # Task configuration
        task_configs = {
            "classification": {
                "task_type": "text_classification",
                "metrics": ["accuracy", "f1"],
                "prompt_template": [
                    {
                        "system": "Classify the security configuration type based on the description.",
                        "user": "Configuration: {input}",
                        "assistant": "Configuration type: {output}"
                    },
                    {
                        "system": "As a security expert, categorize this configuration into its appropriate type.",
                        "user": "Security config:\n{input}",
                        "assistant": "This is a {output} type configuration."
                    },
                    {
                        "system": "You are a security configuration analyst. Determine the configuration category.",
                        "user": "Analyze this config:\n{input}",
                        "assistant": "Based on the description, this falls under the {output} category."
                    }
                ]
            },
            "requirement_extraction": {
                "task_type": "information_extraction",
                "metrics": ["precision", "recall", "f1"],
                "prompt_template": [
                    {
                        "system": "Extract the technical mechanisms and parameters from the security configuration.",
                        "user": "{input}",
                        "assistant": "Technical mechanisms: {output[technical_mechanisms]}\nParameters: {output[parameters]}"
                    },
                    {
                        "system": "As a security analyst, identify and list the technical components and parameters in this configuration.",
                        "user": "Security configuration:\n{input}",
                        "assistant": "I've identified the following:\nTechnical mechanisms: {output[technical_mechanisms]}\nRequired parameters: {output[parameters]}"
                    },
                    {
                        "system": "You are a security configuration expert. Parse and extract the key technical elements.",
                        "user": "Parse this configuration:\n{input}",
                        "assistant": "Configuration analysis:\n- Technical mechanisms: {output[technical_mechanisms]}\n- Configuration parameters: {output[parameters]}"
                    }
                ]
            },
            "reference_prediction": {
                "task_type": "multi_label_classification",
                "metrics": ["precision", "recall", "f1"],
                "prompt_template": [
                    {
                        "system": "Predict relevant security standards and guidelines for the configuration.",
                        "user": "{input}",
                        "assistant": "Relevant standards: {output}"
                    },
                    {
                        "system": "As a security compliance expert, identify applicable security standards for this configuration.",
                        "user": "Configuration details:\n{input}",
                        "assistant": "This configuration is related to the following standards: {output}"
                    },
                    {
                        "system": "You are a security standards specialist. List all relevant security guidelines.",
                        "user": "Review this security config:\n{input}",
                        "assistant": "Based on the configuration, these standards apply: {output}"
                    }
                ]
            },
            "platform_detection": {
                "task_type": "single_label_classification",
                "metrics": ["accuracy"],
                "prompt_template": [
                    {
                        "system": "Determine the platform this security configuration applies to.",
                        "user": "{input}",
                        "assistant": "Platform: {output}"
                    },
                    {
                        "system": "As a system analyst, identify the target platform for this security configuration.",
                        "user": "Configuration:\n{input}",
                        "assistant": "This configuration is designed for {output} platforms."
                    },
                    {
                        "system": "You are a platform compatibility expert. Specify the intended platform.",
                        "user": "Analyze this config:\n{input}",
                        "assistant": "This security configuration is compatible with {output}."
                    }
                ]
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

# Dataset loader registration table
DATASET_LOADERS = {
    "gpqa": GPQALoader,
    "trec": TRECLoader,
    "agnews": AGNewsLoader,
    "lexglue": LexGlueLoader,
    "pubmedqa": PubMedQALoader,
    "cce": CCELoader,
}