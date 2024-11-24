class BrainwashAttack(ICLAttackStrategy):
    def __init__(self, attack_config: Dict[str, Any]):
        super().__init__(attack_config)
        self.max_misleadings = attack_config.get('max_misleadings', 10)
        self.threshold = attack_config.get('brainwash_threshold', 5)
        self.num_wrong_labels = attack_config.get('num_wrong_labels', 3)  # 新增参数

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
                                 attack_sample: Dict[str, str], wrong_label: str) -> int:
        def generate_prompt_and_request(iterations):
            query_prompt = prompt.copy()
            for _ in range(iterations):
                query_prompt = query_prompt + [{
                    "role": "user",
                    "content": self.user_prompt.format(input=attack_sample["input"])
                }, {
                    "role": "assistant",
                    "content": self.assistant_prompt.format(output=wrong_label)
                }]
            query_prompt = query_prompt + [{
                "role": "user",
                # XXX
                "content": self.user_prompt.format(input=attack_sample["input"]) + " Type:"
            }]
            return model.query(query_prompt, "Brainwash Attack")[0]
        
        left, right = 0, self.max_misleadings
        mid = self.max_misleadings
        while left < right:
            old_mid = mid
            response = generate_prompt_and_request(mid)
            if self.brainwashed(response, wrong_label):
                right = mid
            else:
                left = mid + 1
            mid = (left + right) // 2
        final_response = generate_prompt_and_request(old_mid)
        if old_mid != self.max_misleadings:
            while not self.brainwashed(final_response, wrong_label):
                old_mid += 1
                final_response = generate_prompt_and_request(old_mid)
            logger.info(f"Brainwashed to \"{wrong_label}\" in {old_mid} turns: {final_response}")
        else:
            logger.info(f"Failed to brainwash to \"{wrong_label}\" in {old_mid} turns: {final_response}")
        return old_mid

    @ICLAttackStrategy.cache_results
    def attack(self, model: ModelInterface):
        for icl_samples, attack_sample, is_member in tqdm(self.data_loader.test()):
            icl_prompt = self.generate_icl_prompt(icl_samples)
            
            correct_label = attack_sample["output"]
            wrong_labels = [label for label in self.label_translation.values() if label != correct_label]
            selected_wrong_labels = random.sample(wrong_labels, min(self.num_wrong_labels, len(wrong_labels)))

            logger.info(f"Sample: {attack_sample['input']}")
            logger.info(f"Correct label: {correct_label}")

            iterations = []
            for wrong_label in selected_wrong_labels:
                iteration = self.binary_search_iterations(model, icl_prompt, attack_sample, wrong_label)
                iterations.append(iteration)
            
            avg_iterations = np.mean(iterations)
            pred_member = avg_iterations >= self.threshold
            self.results.append((pred_member, is_member, avg_iterations))
            
            # 添加日志输出
            logger.info(f"Iterations: {iterations}")
            logger.info(f"Is member: {is_member}, Predicted member: {pred_member}")
            logger.info("-" * 50)

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
        if self.attack_config.get('plot_roc', False):
            EvaluationMetrics.plot_roc(fpr, tpr, roc_auc, f'roc_curve_{self.__class__.__name__}.png')
        if self.attack_config.get('plot_log_roc', False):
            EvaluationMetrics.plot_log_roc(log_fpr, log_tpr, log_auc, f'log_roc_curve_{self.__class__.__name__}.png')

        return metrics
