from .common import *

from attack import ICLAttackStrategy
from attack.brainwash import BrainwashAttack
from attack.repeat import RepeatAttack

from llm.query import ModelInterface
from utils import SLogger, SDatasetManager

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from transformers import Trainer, TrainingArguments

from sklearn.model_selection import train_test_split

@dataclass
class HybridDataCollator:
    def __init__(self, scalers=None):
        self.scalers = scalers
        
    def __call__(self, features: List[Tuple[float, float, bool]]) -> Dict[str, torch.Tensor]:
        # 分离特征和标签
        brainwash_scores = []
        repeat_scores = []
        labels = []
        
        for brainwash_score, repeat_score, is_member in features:
            brainwash_scores.append(brainwash_score)
            repeat_scores.append(repeat_score)
            labels.append(float(is_member))
        
        # 归一化处理（如果scaler不为None）
        if self.scalers is not None:
            brainwash_scores = self.scalers[0].transform(np.array(brainwash_scores).reshape(-1, 1)).flatten()
            repeat_scores = self.scalers[1].transform(np.array(repeat_scores).reshape(-1, 1)).flatten()
            
        # 将分数组合成特征向量
        features_list = [[b, r] for b, r in zip(brainwash_scores, repeat_scores)]
        
        return {
            'features': torch.tensor(features_list, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }

class HybridModel(nn.Module):
    def __init__(self, hidden_sizes=[64, 32]):
        super().__init__()
        
        layers = []
        # Input layer (2 features: brainwash and repeat scores)
        layers.append(nn.Linear(2, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Dropout(0.3))
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.Dropout(0.3))
            
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, features, labels=None):
        logits = self.model(features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

class HybridAttack(ICLAttackStrategy):
    def __init__(self, attack_config: Dict[str, Any]):
        super().__init__(attack_config)
        
        # Ensure same random seed for both attacks
        attack_config['random_seed'] = self.random_seed
        
        self.brainwash_attack = BrainwashAttack(attack_config)
        self.repeat_attack = RepeatAttack(attack_config)
        self.classifier = HybridModel(hidden_sizes=[64, 32])
        self.classifier.to("cuda")
        
        self.classifier_dir = Path(self.logger.output_dir)/"hybrid-mlp"

        # Training configs
        self.training_args = TrainingArguments(
            output_dir=self.classifier_dir,
            num_train_epochs=attack_config.get('num_epochs', 500),
            per_device_train_batch_size=attack_config.get('batch_size', 32),
            per_device_eval_batch_size=attack_config.get('batch_size', 32),
            learning_rate=attack_config.get('learning_rate', 2e-5),
            weight_decay=attack_config.get('weight_decay', 0.01),
            logging_dir=self.classifier_dir/"logs",
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_steps=50,
            save_total_limit=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )

    def prepare(self, data_config: Dict[str, Any]):
        super().prepare(data_config)
        self.brainwash_attack.prepare(data_config, self.data_loader)
        self.repeat_attack.prepare(data_config, self.data_loader)

    def _attack(self, model: 'ModelInterface'):
        # 执行子攻击并保存结果
        self.brainwash_attack.attack(model)
        self.repeat_attack.attack(model)
        
        # 创建结果表格
        results_table = "dataset_overview"
        self.logger.new_table(results_table)
        
        attack_data = []
        for (bw_pred, bw_true, bw_score), (rp_pred, rp_true, rp_score) in zip(
            self.brainwash_attack.results, self.repeat_attack.results
        ):
            # 验证两个攻击的ground truth一致
            assert bw_true == rp_true, "Inconsistent ground truth between attacks"
            
            # 记录结果
            row = self.logger.new_row(results_table)
            self.logger.add("brainwash_score", bw_score)
            self.logger.add("repeat_score", rp_score)
            self.logger.add("is_member", bw_true)
            
            attack_data.append((bw_score, rp_score, bw_true))
            
        return attack_data

    def plot_attack_scores(self, data):
        plt.figure(figsize=(10, 8))
        
        # 提取分数和标签
        brainwash_scores = np.array([x[0] for x in data])
        repeat_scores = np.array([x[1] for x in data])
        labels = np.array([x[2] for x in data])

        # 绘制散点图
        plt.scatter(brainwash_scores[labels==1], repeat_scores[labels==1], 
                   c='red', label='Member', alpha=0.6)
        plt.scatter(brainwash_scores[labels==0], repeat_scores[labels==0], 
                   c='blue', label='Non-member', alpha=0.6)
        
        plt.xlabel('Brainwash Attack Score (Normalized)')
        plt.ylabel('Repeat Attack Score (Normalized)')
        plt.title('Hybrid Attack: Brainwash vs Repeat Scores')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        self.logger.savefig('hybrid_attack_scores.png')
        plt.close()

    def attack(self, model: 'ModelInterface'):
        # 尝试加载缓存的攻击数据
        self.attack_data = self.logger.load_data("hybrid_attack_data")
        if self.attack_data is None:
            self.attack_data = self._attack(model)
            self.logger.save_data(self.attack_data, "hybrid_attack_data")
            self.logger.save()
        else:
            print("Loaded cached attack data.")
        
        # 划分训练集和验证集
        train_length = self.train_attack
        train_data = self.attack_data[:train_length]
        eval_data = self.attack_data[train_length:]

        # 用全部数据，找一个合适的scaler
        all_features = HybridDataCollator()(self.attack_data)['features']
        brainwash_features = all_features[:, 0].reshape(-1, 1)
        brainwash_scaler = MinMaxScaler().fit(brainwash_features)
        repeat_features = all_features[:, 1].reshape(-1, 1)
        repeat_scaler = MinMaxScaler().fit(repeat_features)
        self.data_collator = HybridDataCollator(scalers=[brainwash_scaler, repeat_scaler])
        
        # 可视化攻击分数分布
        self.plot_attack_scores(self.attack_data)

        # self.logger.save()
        # raise KeyboardInterrupt("Stop here for now.")

        # 检查模型是否已训练
        classifier_path = os.path.join(self.classifier_dir, "model.safetensors")
        if not os.path.exists(classifier_path):
            trainer = Trainer(
                model=self.classifier,
                args=self.training_args,
                data_collator=self.data_collator,
                train_dataset=train_data,
                eval_dataset=eval_data
            )
            
            self.logger.info("Starting model training...")
            trainer.train()
            self.logger.info("Model training completed.")
            trainer.save_model(self.classifier_dir)
        else:
            self.logger.info("Model already trained. Loading the model...")
            from safetensors.torch import load_file
            self.classifier.load_state_dict(load_file(classifier_path))
            
        # 进行预测并记录结果
        self.classifier.eval()
        predictions_table = "predictions"
        self.logger.new_table(predictions_table)
        
        eval_features = self.data_collator(eval_data)
        
        with torch.no_grad():
            predictions = self.classifier(eval_features['features'].to('cuda'))['logits']
            predictions = predictions.cpu().numpy().flatten()
            
            for i, (pred, (bw_score, rp_score, is_member)) in enumerate(zip(predictions, eval_data)):
                pred_member = bool(pred > 0.5)
                self.results.append((pred_member, bool(is_member), float(pred)))
                
                row = self.logger.new_row(predictions_table)
                self.logger.add("sample_id", train_length + i)
                self.logger.add("brainwash_score", bw_score)
                self.logger.add("repeat_score", rp_score)
                self.logger.add("prediction", pred)
                self.logger.add("is_member", is_member)

    def evaluate(self):
        predictions = [bool(pred) for pred, _, _ in self.results]
        ground_truth = [bool(truth) for _, truth, _ in self.results]
        scores = [score for _, _, score in self.results]
        
        metrics = EvaluationMetrics.calculate_advantage(predictions, ground_truth)
        fpr, tpr, roc_auc = EvaluationMetrics.calculate_roc_auc(ground_truth, scores)
        metrics['auc'] = roc_auc
        
        # 记录子攻击的评估结果
        # print(f"Repeat: {self.repeat_attack.evaluate()}")
        # print(f"Brainwash: {self.brainwash_attack.evaluate()}")
        
        # 将评估指标添加到logger
        self.logger.new_table("metrics")
        for key, value in metrics.items():
            self.logger.add(key, value, "metrics")
            
        self.logger.save()
        return metrics
