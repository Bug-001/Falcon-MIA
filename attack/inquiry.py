from .common import *

from . import ICLAttackStrategy

class InquiryAttack(ICLAttackStrategy):
    def __init__(self, attack_config):
        super().__init__(attack_config)
        self.inquiry_template = attack_config.get('inquiry_template', "Have you seen this sentence before: {sample}?")
        self.positive_keywords = attack_config.get('positive_keywords', ["yes", "seen", "encountered", "familiar"])
        self.negative_keywords = attack_config.get('negative_keywords', ["no", "not seen", "unfamiliar"])

    def construct_inquiry(self, sample):
        return self.inquiry_template.format(input=sample)

    def is_member_by_response(self, response):
        words = [self.remove_punctuation(word.lower()) for word in self.remove_punctuation(response).split()]
        print(words)

        if len(words) == 0:
            return None

        # 检查负面关键词
        if any(word in self.negative_keywords for word in words):
            return False
        if "have not seen" in response.lower() or "haven't seen" in response.lower():
            return False
        
        # 检查正面关键词
        if any(word in self.positive_keywords for word in words):
            return True
        if "have seen" in response.lower() or "have encountered" in response.lower():
            return True
        
        if words[0].startswith("1"):
            return True
        elif words[0].startswith("0"):
            return False

        # 模型未给出有效信息
        return None

    @ICLAttackStrategy.cache_results
    def attack(self, model):
        data_loader = self.data_loader.train() + self.data_loader.test()
        for icl_samples, attack_sample, is_member in tqdm(data_loader):
            icl_prompt = self.generate_icl_prompt(icl_samples)
            
            final_prompt = icl_prompt + [{
                "role": "user",
                "content": self.construct_inquiry(attack_sample["input"])
            }]
            response = model.query(final_prompt, "Inquiry Attack")[0]
            
            pred_member = self.is_member_by_response(response)
            if pred_member is not None:
                self.results.append((pred_member, is_member))
            else:
                self.results.append((random.random() < 0.5, is_member))
            
            # 添加日志输出
            self.logger.info(f"Sample: {attack_sample['input']}")
            self.logger.info(f"Final Prompt: {final_prompt}")
            self.logger.info(f"Model response: {response}")
            self.logger.info(f"Is member: {is_member}, Predicted member: {pred_member}")
            self.logger.info("-" * 50)

    def evaluate(self):
        predictions = [bool(pred) for pred, _ in self.results]
        ground_truth = [bool(truth) for _, truth in self.results]
        metrics = EvaluationMetrics.calculate_advantage(predictions, ground_truth)
        # 计算ROC曲线和AUC
        fpr, tpr, roc_auc = EvaluationMetrics.calculate_roc_auc(ground_truth, predictions)
        metrics['auc'] = roc_auc

        # 计算log ROC
        log_fpr, log_tpr, log_auc = EvaluationMetrics.calculate_log_roc_auc(ground_truth, predictions)

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