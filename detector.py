import os
import json
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModel, PreTrainedModel, PretrainedConfig
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from datasets import Dataset
from collections import defaultdict

@dataclass
class MembershipDataCollator:
    tokenizer: AutoTokenizer
    max_length: int = 512
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        
        # 编码原始文本
        orig_texts = [f['original_text'] for f in features]
        orig_encodings = self.tokenizer(
            orig_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        if isinstance(features[0]['responses'], str):
            # 批量处理response
            responses = [f['responses'] for f in features]
            resp_encodings = self.tokenizer(
                responses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # 准备标签和level
            labels = torch.tensor([f['membership'] for f in features], dtype=torch.float32)
            levels = torch.tensor([f['levels'] for f in features], dtype=torch.float32)

        else:
            # 获取当前批次中最大的level数量
            max_levels = max(len(f['responses']) for f in features)
            
            # 为每个level创建编码
            resp_encodings = {
                'input_ids': [],
                'attention_mask': [],
            }

            # 将response进行编码
            resp_list = []
            for feature in features:
                resp = feature['responses'] + [''] * (max_levels - len(feature['responses']))
                resp_list.extend(resp)
            encodings = self.tokenizer(
                resp_list,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            resp_encodings['input_ids'] = encodings['input_ids'].reshape(batch_size, max_levels, -1).flatten(1, 2)
            resp_encodings['attention_mask'] = encodings['attention_mask'].reshape(batch_size, max_levels, -1).flatten(1, 2)
            
            # 准备level值和标签
            levels = torch.zeros(batch_size, max_levels, dtype=torch.float32)
            for i, feature in enumerate(features):
                levels[i, :len(feature['levels'])] = torch.tensor(feature['levels'])
            
            labels = torch.tensor([f['membership'] for f in features], dtype=torch.float32)
        
        return {
            'orig_input_ids': orig_encodings['input_ids'],
            'orig_attention_mask': orig_encodings['attention_mask'],
            'resp_input_ids': resp_encodings['input_ids'],
            'resp_attention_mask': resp_encodings['attention_mask'],
            'levels': levels,
            'labels': labels,
        }

class MembershipDataset():
    def __init__(self, data):
        # 按sample_id对数据进行分组
        grouped_data = defaultdict(list)
        for item in data:
            key = (item['sample_id'], item['original_text'])  # 使用sample_id和original_text作为key
            grouped_data[key].append(item)
        
        # 重新组织数据
        self.data = []
        for (sample_id, original_text), group in grouped_data.items():
            # 按level排序
            group.sort(key=lambda x: x['level'])
            
            # 现在让不同level成为不同训练例子
            for item in group:
                self.data.append({
                    'sample_id': sample_id,
                    'original_text': original_text,
                    'responses': item['response'],
                    'levels': item['level'],
                    'membership': group[0]['membership']  # membership应该对同一个sample_id是一致的
                })

            # self.data.append({
            #     'sample_id': sample_id,
            #     'original_text': original_text,
            #     'responses': [item['response'] for item in group],
            #     'levels': [item['level'] for item in group],
            #     'membership': group[0]['membership']  # membership应该对同一个sample_id是一致的
            # })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, _ = self.attention(query, key, value, key_padding_mask=key_padding_mask)
        return self.norm(query + attn_output)

class MembershipDetectionModel(nn.Module):
    def __init__(self, model_name='microsoft/deberta-v3-large'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.cross_attention = CrossAttention(hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        orig_input_ids,
        orig_attention_mask,
        resp_input_ids,
        resp_attention_mask,
        levels=None,
        labels=None,
    ):
        # 编码original_text
        orig_outputs = self.encoder(
            input_ids=orig_input_ids,
            attention_mask=orig_attention_mask
        )
        orig_hidden = orig_outputs.last_hidden_state
        
        # 编码拼接后的responses
        resp_outputs = self.encoder(
            input_ids=resp_input_ids,
            attention_mask=resp_attention_mask
        )
        resp_hidden = resp_outputs.last_hidden_state
        
        # 交叉注意力
        attended_orig = self.cross_attention(
            orig_hidden,
            resp_hidden,
            resp_hidden,
            key_padding_mask=~resp_attention_mask.bool()
        )
        
        attended_resp = self.cross_attention(
            resp_hidden,
            orig_hidden,
            orig_hidden,
            key_padding_mask=~orig_attention_mask.bool()
        )
        
        # 池化
        orig_pooled = attended_orig.mean(dim=1)
        resp_pooled = attended_resp.mean(dim=1)
        
        # 拼接并分类
        concat = torch.cat([orig_pooled, resp_pooled], dim=-1)
        logits = self.classifier(concat).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}
    
class MembershipInference:
    def __init__(self, model_path, encoder_name='microsoft/deberta-v3-large', device='cuda' if torch.cuda.is_available() else 'cpu'):
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
        self.model = MembershipDetectionModel(encoder_name)
        self.model.load_state_dict(state_dict)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        
        self.model.eval()
        self.device = device
        self.model.to(device)
        
        self.collator = MembershipDataCollator(self.tokenizer)
    
    def predict(self, original_text, response, threshold=0.5):
        batch = self.collator.encode(original_text, response)
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            prob = outputs['logits'].cpu().numpy()[0]
            
        return {
            'probability': float(prob),
            'is_member': bool(prob > threshold),
        }
    
    # def predict_batch(self, samples, threshold=0.5):
    #     features = [
    #         {
    #             'original_text': sample['original_text'],
    #             'response': sample['response']
    #         }
    #         for sample in samples
    #     ]
        
    #     batch = self.collator(features)
    #     inputs = {k: v.to(self.device) for k, v in batch.items()}
        
    #     with torch.no_grad():
    #         outputs = self.model(**inputs)
    #         probs = outputs['logits'].cpu().numpy()
            
    #     return [
    #         {
    #             'probability': float(prob),
    #             'is_member': bool(prob > threshold),
    #         }
    #         for prob in probs
    #     ]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (logits > 0.5).astype(np.int64)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    encoder_name = 'microsoft/deberta-v3-large'

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    
    # 准备数据集
    data_path = "cache/data/"
    with open(os.path.join(data_path, 'Meta-Llama-3-8B-Instruct--output.json'), 'r') as f:
        data = json.load(f)
    data_len = len(data)
    train_dataset = MembershipDataset(data[:int(data_len*0.8)])  # train_data需要自行加载
    eval_dataset = MembershipDataset(data[int(data_len*0.8):])    # eval_data需要自行加载
    
    # 创建数据整理器
    data_collator = MembershipDataCollator(tokenizer=tokenizer)
    
    # 创建模型
    model = MembershipDetectionModel(encoder_name)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="cache/model/detector/llama3-test",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=10,
        learning_rate=2e-5,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        num_train_epochs=30,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        remove_unused_columns=False,
        report_to="wandb",          # 启用wandb记录
        logging_dir="cache/model/detector/log",
        logging_steps=10,
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存最终模型
    trainer.save_model("cache/model/detector/llama3-test")

if __name__ == "__main__":
    main()