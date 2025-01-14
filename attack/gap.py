from .common import *

from llm.query import ModelInterface
from attack import ICLAttackStrategy
from typing import Dict


from string_utils import StringHelper

class GAPAttack(ICLAttackStrategy):
    def __init__(self, attack_config):
        super().__init__(attack_config)
        self.similarity_threshold = attack_config.get('similarity_threshold', 0.8)
        self.shelper = StringHelper()
        
    def _calculate_similarity(self, response: str, true_output: str) -> float:
        """辅助方法：计算相似度"""
        response_embedding = self.shelper.preprocess_text(response, 'semantic')
        true_embedding = self.shelper.preprocess_text(true_output, 'semantic')
        return self.shelper.semantic_cosine_similarity(response_embedding, true_embedding)

    def is_member_by_response(self, response: str, true_output: str) -> Tuple[float, bool]:
        # 首先检查是否包含重复/已见过的提示词
        duplicate_phrases = ["already", "duplicate", "same", "again", "previously", "repeat"]
        for phrase in duplicate_phrases:
            if phrase in response.lower():
                return 1.0, True
        
        # 计算响应和真实输出之间的语义相似度
        similarity = self._calculate_similarity(response, true_output)
        return similarity, (similarity >= self.similarity_threshold)

    @ICLAttackStrategy.cache_results
    def attack(self, model: 'ModelInterface'):
        self.logger.new_table("gap-attack_results")
        
        data_loader = self.data_loader.train() + self.data_loader.test()

        template = self.train_template
        # template = self.data_config['prompt_template']
        
        for icl_samples, attack_sample, is_member in tqdm(data_loader):
            icl_prompt = self.generate_icl_prompt(icl_samples)

            final_prompt = icl_prompt + [{
                "role": "user",
                "content": template['user'].format(input=attack_sample["input"])
            }]
            response = model.query(final_prompt, "Question Classification")[0]
            # We assume that assistant is easy to be detected
            true_response = template['assistant'].format(output=attack_sample["output"])

            # 计算相似度并记录结果
            similarity, pred_member = self.is_member_by_response(response, true_response)
            self.results.append((pred_member, is_member, similarity))

            # 记录到表格中
            self.logger.new_row("gap-attack_results")
            self.logger.add("Input", attack_sample["input"])
            self.logger.add("True Response", true_response)
            self.logger.add("Generated", response)
            self.logger.add("Similarity", similarity)
            self.logger.add("Is member", is_member)
            self.logger.add("Predicted member", pred_member)
            self.logger.info(final_prompt)

            self.logger.info("-" * 50)
        
        self.logger.save()

    def evaluate(self) -> Dict[str, float]:
        predictions = [bool(pred) for pred, _, _ in self.results]
        ground_truth = [bool(truth) for _, truth, _ in self.results]
        similarities = [sim for _, _, sim in self.results]
        
        metrics = EvaluationMetrics.calculate_advantage(predictions, ground_truth)
        
        # 计算ROC曲线和AUC
        fpr, tpr, roc_auc = EvaluationMetrics.calculate_roc_auc(ground_truth, similarities)
        metrics['auc'] = roc_auc
        
        # 计算log ROC
        log_fpr, log_tpr, log_auc = EvaluationMetrics.calculate_log_roc_auc(ground_truth, similarities)
        
        # 存储ROC和log ROC数据
        metrics.update({
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'log_fpr': log_fpr,
            'log_tpr': log_tpr,
            'log_auc': log_auc
        })
        
        self.logger.save_json('metrics.json', metrics)
        self.logger.info(f"Metrics: {metrics}")
        return metrics