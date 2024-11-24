from . import ICLAttackStrategy
from typing import Dict

from ..utils import EvaluationMetrics

class GAPAttack(ICLAttackStrategy):
    def is_member_by_response(self, response: str, true_label: str) -> bool:
        # 第一种方式：判断模型是否见过输入
        duplicate_phrases = ["already", "duplicate", "same", "again", "previously", "repeat"]
        for phrase in duplicate_phrases:
            if phrase in response.lower():
                return True

        # 获取所有可能的标签
        all_labels = set(self.label_translation.values())
        
        # 如果没有翻译字典，就使用原始的输出字段值
        if not all_labels:
            all_labels = set(self.dataset['train'][self.output_field] + self.dataset['test'][self.output_field])
        
        # 检查真实标签是否在响应中，且其他标签都不在响应中
        true_label_in_response = true_label in response
        other_labels_not_in_response = all(label not in response for label in all_labels if label != true_label)
        
        return true_label_in_response and other_labels_not_in_response

    @ICLAttackStrategy.cache_results
    def attack(self, model: 'ModelInterface'):
        for icl_samples, attack_sample, is_member in tqdm(self.data_loader.test()):
            icl_prompt = self.generate_icl_prompt(icl_samples)

            final_prompt = icl_prompt + [{
                "role": "user",
                "content": self.user_prompt.format(input=attack_sample["input"])
            }]
            response = model.query(final_prompt, "Question Classification")[0]
            pred_member = self.is_member_by_response(response, str(attack_sample["output"]))
            self.results.append((pred_member, is_member))

            # 添加日志输出
            logger.info(f"Input: {attack_sample['input']}")
            logger.info(f"True label: {attack_sample['output']}")
            logger.info(f"Model response: {response}")
            logger.info(f"Is member: {is_member}, Predicted member: {pred_member}")
            logger.info("-" * 50)

    def evaluate(self) -> Dict[str, float]:
        predictions = [bool(pred) for pred, _ in self.results]
        ground_truth = [bool(truth) for _, truth in self.results]
        return EvaluationMetrics.calculate_advantage(predictions, ground_truth)