import argparse
import yaml
from types import SimpleNamespace
import os
from dotenv import load_dotenv

from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoConfig
from accelerate import Accelerator
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PrefixTuningConfig, PromptEncoderConfig, IA3Config
from transformers import LlamaTokenizer

from tools.data_preprocessing import preprocess
# from remote_model_api import RemoteModelAPI
from tools.utils import get_logger, print_trainable_parameters

logger = get_logger("finetune", "info")

def load_config(config_path):

    def convert_scientific_notation(data):
        if isinstance(data, dict):
            return {k: convert_scientific_notation(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_scientific_notation(i) for i in data]
        elif isinstance(data, str):
            try:
                return float(data)
            except ValueError:
                return data
        return data
    
    def dict_to_namespace(d):
        """递归地将字典转换为 SimpleNamespace"""
        if isinstance(d, dict):
            for key, value in d.items():
                d[key] = dict_to_namespace(value)
            return SimpleNamespace(**d)
        elif isinstance(d, list):
            return [dict_to_namespace(item) for item in d]
        else:
            return d

    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    config_dict = convert_scientific_notation(config_dict)
    
    # 确保 data 配置中包含 cache_enabled 选项
    if 'data' in config_dict and 'cache_enabled' not in config_dict['data']:
        config_dict['data']['cache_enabled'] = True  # 默认启用缓存

    return dict_to_namespace(config_dict)

def load_env_variables():
    load_dotenv()
    os.environ['HTTP_PROXY'] = os.getenv('HTTP_PROXY', '')
    os.environ['HTTPS_PROXY'] = os.getenv('HTTPS_PROXY', '')

def initialize_tokenizer(model_config, misc_config):
    if model_config.type == "transformers":
        model_type = AutoConfig.from_pretrained(model_config.name).model_type
        tokenizer_class = LlamaTokenizer if model_type == "llama" else AutoTokenizer
        tokenizer = tokenizer_class.from_pretrained(
            model_config.name, 
            token=misc_config.token or os.getenv("HF_TOKEN", ""),
            trust_remote_code=model_config.trust_remote_code, 
            cache_dir=model_config.hf_cache_dir,  # 使用HuggingFace缓存目录
            add_eos_token=model_config.tokenizer.add_eos_token, 
            add_bos_token=model_config.tokenizer.add_bos_token,
            use_fast=True
        )
        if model_config.tokenizer.pad_token_id is not None:
            tokenizer.pad_token_id = model_config.tokenizer.pad_token_id
        elif tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
    elif model_config.type == "remote":
        raise NotImplementedError("Remote model tokenization is not supported yet")

def initialize_model(model_config, peft_config, hardware_config, misc_config):
    if model_config.type == "transformers":
        model_config_obj = AutoConfig.from_pretrained(model_config.name, cache_dir=model_config.hf_cache_dir)
        model_config_obj.use_cache = False

        if model_config.use_flash_attention and model_config_obj.model_type == "llama" and torch.cuda.get_device_capability()[0] >= 8:
            from llm.tools.llama_patch import replace_attn_with_flash_attn
            replace_attn_with_flash_attn()

        kwargs = {"device_map": "auto"} if hardware_config.split_model else {"device_map": None}

        bnb_config = None
        if model_config.quantization.use_int4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif model_config.quantization.use_int8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_config.name, 
            token=misc_config.token or os.getenv("HF_TOKEN", ""),
            quantization_config=bnb_config,
            trust_remote_code=model_config.trust_remote_code, 
            cache_dir=model_config.hf_cache_dir,  # 使用HuggingFace缓存目录
            torch_dtype=torch_dtype, 
            config=model_config_obj, 
            **kwargs
        )

        if not peft_config.disable_peft:
            model = apply_peft(peft_config, model)

        return model
    elif model_config.type == "remote":
        raise NotImplementedError("Remote model training is not supported yet")
        # return RemoteModelAPI(model_config)

def apply_peft(peft_config, model):
    if peft_config.method == "lora":
        peft_config_obj = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=peft_config.lora.rank,
            lora_alpha=peft_config.lora.alpha,
            lora_dropout=peft_config.lora.dropout
        )
    elif peft_config.method == "prefix-tuning":
        peft_config_obj = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            num_virtual_tokens=peft_config.prefix_tuning.num_virtual_tokens,
            encoder_hidden_size=peft_config.prefix_tuning.encoder_hidden_size)
    elif peft_config.method == "p-tuning":
        peft_config_obj = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_config.p_tuning.num_virtual_tokens,
            encoder_hidden_size=peft_config.p_tuning.encoder_hidden_size)
    elif peft_config.method == "ia3":
        peft_config_obj = IA3Config(
            peft_type="IA3",
            task_type=TaskType.CAUSAL_LM,
            target_modules=peft_config.ia3.target_modules,
            feedforward_modules=peft_config.ia3.feedforward_modules,
        )
    
    return get_peft_model(model, peft_config_obj)

def get_training_arguments(training_config):
    return TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir=training_config.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy=training_config.evaluation_strategy,
        save_strategy=training_config.save_strategy,
        logging_strategy="steps",
        num_train_epochs=training_config.num_train_epochs,
        eval_steps=training_config.eval_steps,
        save_steps=training_config.save_steps,
        logging_steps=training_config.logging_steps,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_train_batch_size * 2,
        learning_rate=training_config.learning_rate,
        lr_scheduler_type=training_config.lr_scheduler_type,
        warmup_steps=training_config.warmup_steps,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        gradient_checkpointing=training_config.gradient_checkpointing,
        weight_decay=training_config.weight_decay,
        adam_epsilon=1e-6,
        report_to="wandb",
        load_best_model_at_end=False,
        save_total_limit=training_config.save_total_limit,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )

def save_model(model, tokenizer, output_dir):
    # 如果是PEFT模型，先合并基础模型和PEFT权重
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    
    # 保存模型
    model.save_pretrained(output_dir)
    
    # 保存tokenizer
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Model and tokenizer saved to {output_dir}")

def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(examples["x"], truncation=True, padding="max_length", max_length=max_length)

def main(config_path):
    config = load_config(config_path)
    load_env_variables()

    accelerator = Accelerator()

    # 创建实验目录
    experiment_dir = os.path.join(config.training.output_dir, config.id)
    os.makedirs(experiment_dir, exist_ok=True)

    # 检查实验是否已完成
    if os.path.exists(os.path.join(experiment_dir, "config.json")):
        if accelerator.is_main_process:
            logger.info(f"Experiment {config.id} has already been completed.")
        accelerator.wait_for_everyone()
        return

    tokenizer = initialize_tokenizer(config.model, config.miscellaneous)
    model = initialize_model(config.model, config.peft, config.hardware, config.miscellaneous)
    print_trainable_parameters(model)

    with accelerator.main_process_first():
        # 直接调用 preprocess 函数，获取已经处理好的数据集
        train_dataset, valid_dataset, _ = preprocess(config.data)

        # 对数据集进行 tokenize
        train_dataset = train_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, config.data.block_size),
            batched=True
        )
        valid_dataset = valid_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, config.data.block_size),
            batched=True
        )

    logger.info(f"Training with {Accelerator().num_processes} GPUs")
    training_args = get_training_arguments(config.training)

    if config.model.type == "transformers":
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            dataset_text_field="text",
            tokenizer=tokenizer,
        )
        trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
        
        # 保存微调后的模型
        save_model(model, tokenizer, os.path.join(config.training.output_dir))
    elif config.model.type == "remote":
        model.train(train_dataset, valid_dataset, training_args)
        # 对于远程模型，可能需要特殊的保存方法
        model.save(os.path.join(config.training.output_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, 
                        default=os.path.join(os.path.dirname(__file__), "examples", "fine-tune.yaml"),
                        help="Path to the YAML config file")
    args = parser.parse_args()
    main(args.config)