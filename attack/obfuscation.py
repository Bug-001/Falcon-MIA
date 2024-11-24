from .common import *

from attack import ICLAttackStrategy
from attack.string_utils import ObfuscationTechniques, StringHelper

import torch
from torch import nn
from transformers import Trainer, TrainingArguments

@dataclass
class ObfuscationDataCollator:
    def __call__(self, features: List[Tuple[Dict, bool]]) -> Dict[str, torch.Tensor]:
        # 分离特征和标签
        features_list = []
        labels = []
        for data_point, label in features:
            # 获取所有unique的similarity类型
            similarity_types = sorted(list(set(
                sim_type for level_data in data_point.values() 
                for sim_type in level_data.keys()
            )))
            
            # 构建特征向量
            feature_vector = []
            for level, sim_dict in sorted(data_point.items()):
                feature_vector.append(level)  # 添加level值
                feature_vector.extend(sim_dict.get(sim_type, 0.0) for sim_type in similarity_types)
            
            features_list.append(feature_vector)
            labels.append(float(label))

        return {
            'features': torch.tensor(features_list, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }

class ObfuscationModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32]):
        super().__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
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

class ObfuscationAttack(ICLAttackStrategy):
    def __init__(self, attack_config: Dict[str, Any]):
        super().__init__(attack_config)
        self.obfuscator = ObfuscationTechniques(attack_config.get('obfuscation_config', {}))
        self.num_obfuscation_levels = attack_config.get('num_obfuscation_levels', 5)
        self.max_obfuscation_level = attack_config.get('max_obfuscation_level', 1)
        self.attack_template = attack_config.get('obsfucation_attack_template', "Classify the following text: {sample}")

        self.shelper = StringHelper()
        self.num_similarities = attack_config.get('num_similarities', 5)
        self.classifier = ObfuscationModel(self.num_obfuscation_levels * (self.num_similarities + 1), hidden_sizes=[64, 32])
        self.classifier.to("cuda")

        self.detector_dir = Path(self.logger.output_dir)/"obf-mlp"
        # Training configs
        self.training_args = TrainingArguments(
            output_dir=self.detector_dir,
            num_train_epochs=attack_config.get('num_epochs', 1000),
            per_device_train_batch_size=attack_config.get('batch_size', 32),
            per_device_eval_batch_size=attack_config.get('batch_size', 32),
            learning_rate=attack_config.get('learning_rate', 2e-5),
            weight_decay=attack_config.get('weight_decay', 0.01),
            logging_dir=self.detector_dir/"logs",
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            eval_steps=50,
            save_steps=50,
            save_total_limit=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )

    def _attack(self, model):
        # 创建主表格和详细记录表格
        results_table = "dataset_overview"
        details_table = "level_details"

        similarities_data = []
        
        self.logger.new_table(results_table)
        self.logger.new_table(details_table)

        train_length = self.attack_config['train_attack']
        data_loader = self.data_loader.train() + self.data_loader.test()
        for i, (icl_samples, attack_sample, is_member) in enumerate(tqdm(data_loader)):
            # 为主表格创建新行
            main_row = self.logger.new_row(results_table)
            
            icl_prompt = self.generate_icl_prompt(icl_samples)
            
            # XXX: 从input和output中选择词数更长的样本，截断，用于混淆
            # 可以考虑采用更加tricky虽然没什么卵用的方式
            input_text = attack_sample["input"].split()
            output_text = attack_sample["output"].split()
            original_text = input_text if len(input_text) > len(output_text) else output_text
            # 截取前50词，不够就全取
            original_text = " ".join(original_text[:100])

            # 记录并打印基本信息
            self.logger.add("sample_id", main_row)
            self.logger.add("type", "train" if i < train_length else "test")
            self.logger.add("Original", original_text)
            self.logger.add("Membership", is_member)
            
            all_level_similarities = dict()
            # 处理每个混淆级别
            for level in np.linspace(0.6, self.max_obfuscation_level, self.num_obfuscation_levels):
                level_row = self.logger.new_row(details_table)
                
                obfuscated_text = self.obfuscator.obfuscate(original_text, level=level)
                query_prompt = icl_prompt + [{
                    "role": "user",
                    "content": self.attack_template.format(input=obfuscated_text)
                }]
                response = model.query(query_prompt, "Obfuscation Attack")[0]
                # response = response.split('\n')[0]  # 只使用第一行
                similarities = self.shelper.calculate_overall_similarity_dict(original_text, response)
                
                # 记录level的详细信息
                self.logger.add("sample_id", main_row)
                self.logger.add("level", level)
                self.logger.add("obfuscated_text", obfuscated_text)
                self.logger.add("response", response)
                self.logger.add("similarities", similarities)
                
                all_level_similarities[level] = similarities

            similarities_data.append((all_level_similarities, is_member))

        return similarities_data

    def attack(self, model):
        self.similarities_data = self.logger.load_data("similarities_data")
        if self.similarities_data == None:
            self.similarities_data = self._attack(model)
            self.logger.save_data(self.similarities_data, "similarities_data")
            self.logger.save()
        else:
            print("Loaded cached similarities data.")

        # Train the model
        # 检查模型是否已经训练好
        classfier_path = os.path.join(self.detector_dir, "model.safetensors")
        if not os.path.exists(classfier_path):
            num_train = self.attack_config['train_attack']
            train_data = self.similarities_data[:num_train]
            test_data = self.similarities_data[num_train:]
            trainer = Trainer(
                model=self.classifier,
                args=self.training_args,
                data_collator=ObfuscationDataCollator(),
                train_dataset=train_data,
                eval_dataset=test_data
            )
            # 训练并保存模型
            self.logger.info("Starting model training...")
            trainer.train()
            self.logger.info("Model training completed.")
            trainer.save_model(self.detector_dir)
        else:
            self.logger.info("Model already trained. Loading the model...")
            from safetensors.torch import load_file
            self.classifier.load_state_dict(load_file(classfier_path))

        # # 找到最佳阈值并更新结果
        # attack_results = self.logger.get_table("attack_results-train")
        # scores = attack_results["mean_similarity"].tolist()
        # true_labels = attack_results["Membership"].tolist()

        # best_threshold, best_f1 = EvaluationMetrics.get_best_threshold(true_labels, scores)
        # self.threshold = best_threshold

        # # 计算准确率
        # accuracy = np.mean((scores >= best_threshold) == true_labels)
        
        # # 更新预测结果
        # attack_results["final_prediction"] = attack_results["mean_similarity"] >= best_threshold

        # # 记录阈值相关的指标
        # self.logger.new_table("metrics")
        # self.logger.new_row("metrics")
        # self.logger.add("best_threshold", best_threshold)
        # self.logger.add("best_f1-train", best_f1)
        # self.logger.add("accuracy-train", accuracy)

    def __add_evaluation_record_to_table(self, table, sample_id, original_text, responses, is_member, similarities, prediction):
        row = self.logger.new_row(table)
        self.logger.add("sample_id", sample_id, table, show=False)
        self.logger.add("Original", original_text, table, show=False)
        self.logger.add("Responses", responses, table, show=False)
        self.logger.add("Membership", is_member, table, show=False)
        self.logger.add("Similarities", similarities, table, show=False)
        self.logger.add("Prediction", prediction, table, show=False)

    def evaluate(self):
        mispredict_table = "mispredicted_results"
        predictions_table = "predictions"
        self.logger.new_table(mispredict_table)
        self.logger.new_table(predictions_table)

        # test_data对应于similariteis_data的最后若干数据，直接对应即可
        num_train = self.attack_config['train_attack']
        test_data = self.logger.get_table("dataset_overview").iloc[num_train:]
        assert test_data['type'].tolist() == ['test'] * len(test_data), "Test data type is not 'test'"
        level_info = self.logger.get_table("level_details").iloc[num_train:]

        # 将similarities_data的test部分经过collator处理后，传入网络中
        self.classifier.eval()
        test_similarities = self.similarities_data[num_train:]
        test_features = ObfuscationDataCollator()(test_similarities)
        
        predictions_values = []
        for i, (feature_vector, is_member) in enumerate(zip(test_features['features'], test_features['labels'])):
            # 将特征向量输入到训练好的模型中
            with torch.no_grad():
                pred = self.classifier(feature_vector.to('cuda').unsqueeze(0))['logits'][0].to('cpu').item()
            predictions_values.append(pred)

            is_member = bool(is_member)
            original_text = test_data.iloc[i]['Original']
            # 获取level_data中sample_id为train_num+i的所有数据，取其Response
            responses = level_info[level_info['sample_id'] == num_train+i]['response'].tolist()
            all_level_similarities = test_similarities[i][0]
            self.__add_evaluation_record_to_table(predictions_table, num_train+i, original_text, responses, is_member, all_level_similarities, pred)
            # 如果预测错误，将其计入到mispredict_table中
            if (pred > 0.5) != is_member:
                self.__add_evaluation_record_to_table(mispredict_table, num_train+i, original_text, responses, is_member, all_level_similarities, pred)

        # 从logger获取最终结果
        predictions = list(map(lambda x: x > 0.5, predictions_values))
        ground_truth = test_features['labels'].tolist()
        
        metrics = EvaluationMetrics.calculate_advantage(predictions, ground_truth)

        # 根据pred_similarity，计算ROC曲线和AUC
        fpr, tpr, roc_auc = EvaluationMetrics.calculate_roc_auc(ground_truth, predictions_values)
        metrics['auc'] = roc_auc

        # 保存metrics
        self.logger.new_table("metrics")
        for key, value in metrics.items():
            self.logger.add(key, value, "metrics")

        # 保存所有结果
        self.logger.save()
        return metrics