from .common import *

from attack import ICLAttackStrategy

from llm.query import ModelInterface
from utils import SLogger, SDatasetManager

import openai

class BrainwashAttack(ICLAttackStrategy):
    def __init__(self, attack_config: Dict[str, Any]):
        super().__init__(attack_config)
        self.max_misleadings = attack_config.get('max_misleadings', 10)
        self.threshold = attack_config.get('brainwash_threshold', 5)
        self.num_wrong_labels = attack_config.get('num_wrong_labels', 3)  # 新增参数
        self.logger.new_table("brainwash-attack_results")  # Initialize table at start

    def brainwashed(self, response: str, wrong_label: str) -> bool:
        # 获取所有可能的标签
        other_labels = {label.lower() for label in self.label_translation.values()}
        wrong_label = wrong_label.lower()
        other_labels.remove(wrong_label)

        # 断句
        words = [self.remove_punctuation(word.lower()) for word in response.split()]
        
        # 检查错误标签是否在响应中，且其他标签都不在响应中
        wrong_label_in_response = any(wrong_label in word for word in words)
        other_labels_not_in_response = all(all(label not in word for word in words) for label in other_labels)
        
        return wrong_label_in_response and other_labels_not_in_response

    def binary_search_iterations(self, model: ModelInterface, prompt: List[Dict[str, str]], 
                                 attack_sample: Dict[str, str], wrong_label: str, is_train: bool) -> int:
        template = self.train_template if is_train else self.test_template

        def generate_prompt_and_request(iterations):
            query_prompt = prompt.copy()
            for _ in range(iterations):
                query_prompt = query_prompt + [{
                    "role": "user",
                    "content": template['user'].format(input=attack_sample["input"])
                }, {
                    "role": "assistant",
                    "content": template['assistant'].format(output=wrong_label)
                }]
            query_prompt = query_prompt + [{
                "role": "user",
                # XXX
                "content": template['user'].format(input=attack_sample["input"]) + " Type:"
            }]
            response = model.query(query_prompt, "Brainwash Attack")[0].split()[0]
            return response
        
        max_misleadings = self.max_misleadings
        left, right = 0, max_misleadings
        mid = max_misleadings
        while left < right:
            old_mid = mid
            try:
                response = generate_prompt_and_request(mid)
                brainwashed = self.brainwashed(response, wrong_label)
                if brainwashed:
                    right = mid
                else:
                    left = mid + 1
            except openai.BadRequestError:
                right = mid - 1
            mid = (left + right) // 2
        return old_mid

    @ICLAttackStrategy.cache_results
    def attack(self, model: ModelInterface):
        self.label_translation = self.sdm.get_config()['label_map']

        train_data = self.data_loader.train()
        test_data = self.data_loader.test()
        data_list = train_data + test_data
        train_length = len(train_data)
        
        for i, (icl_samples, attack_sample, is_member) in enumerate(tqdm(data_list)):
            self.logger.new_row("brainwash-attack_results")
            # 根据索引判断是否是训练数据
            is_train = i < train_length
            icl_prompt = self.generate_icl_prompt(icl_samples, is_train=is_train)
            
            correct_label = attack_sample["output"]
            wrong_labels = [label for label in self.label_translation.values() if label != correct_label]
            selected_wrong_labels = random.sample(wrong_labels, min(self.num_wrong_labels, len(wrong_labels)))

            self.logger.info(f"Sample: {attack_sample['input']}")
            self.logger.info(f"Correct label: {correct_label}")

            iterations = []
            for wrong_label in selected_wrong_labels:
                iteration = self.binary_search_iterations(model, icl_prompt, attack_sample, wrong_label, is_train=is_train)
                iterations.append(iteration)
            
            avg_iterations = np.mean(iterations)
            pred_member = avg_iterations >= self.threshold
            self.results.append((pred_member, is_member, avg_iterations))

            self.logger.add("Sample", attack_sample["input"])
            self.logger.add("Correct Label", correct_label)
            self.logger.add("Wrong Labels", selected_wrong_labels)
            self.logger.add("Iterations", iterations)
            self.logger.add("Average Iterations", avg_iterations)
            self.logger.add("Is member", is_member)
            self.logger.add("Predicted member", pred_member)
            self.logger.info("-" * 50)
        
        self.logger.save()

    def evaluate(self) -> Dict[str, float]:
        predictions = [bool(pred) for pred, _, _ in self.results]
        ground_truth = [bool(truth) for _, truth, _ in self.results]
        similarities = [sim for _, _, sim in self.results]
        
        metrics = EvaluationMetrics.calculate_advantage(predictions, ground_truth)
        metrics['average_similarity'] = np.mean(similarities)
        
        # 计算ROC曲线和AUC
        fpr, tpr, roc_auc = EvaluationMetrics.calculate_roc_auc(ground_truth, similarities)
        metrics['auc'] = roc_auc

        # 计算log ROC
        log_fpr, log_tpr, log_auc = EvaluationMetrics.calculate_log_roc_auc(ground_truth, similarities)
        metrics['log_auc'] = log_auc

        # 存储ROC和log ROC数据
        self.roc_data = {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'log_fpr': log_fpr,
            'log_tpr': log_tpr,
            'log_auc': log_auc
        }

        # 绘制ROC和log ROC曲线
        # if self.attack_config.get('plot_roc', False):
        #     EvaluationMetrics.plot_roc(fpr, tpr, roc_auc, f'roc_curve_{self.__class__.__name__}.png')
        # if self.attack_config.get('plot_log_roc', False):
        #     EvaluationMetrics.plot_log_roc(log_fpr, log_tpr, log_auc, f'log_roc_curve_{self.__class__.__name__}.png')

        self.logger.save_json('metrics.json', metrics)

        return metrics
