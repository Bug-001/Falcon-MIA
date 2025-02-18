from .common import *

from llm.query import ModelInterface
from attack import ICLAttackStrategy
from typing import Dict


from string_utils import StringHelper

class GAPAttack(ICLAttackStrategy):
    def __init__(self, attack_config):
        super().__init__(attack_config)
        self.num_cross_validation = attack_config.get('cross_validation', 1)
        self.shelper = StringHelper()
        # 存储每一折的最优阈值和结果
        self.thresholds = []
        self.fold_results = []
        
    def _calculate_similarity(self, response: str, true_output: str) -> float:
        """辅助方法：计算相似度"""
        response_embedding = self.shelper.preprocess_text(response, 'semantic')
        true_embedding = self.shelper.preprocess_text(true_output, 'semantic')
        return self.shelper.semantic_cosine_similarity(response_embedding, true_embedding)

    @ICLAttackStrategy.cache_results
    def attack(self, model: 'ModelInterface'):
        # 查找results.pkl是否已经存在
        self.results = self.logger.load_data("results.pkl")
        if self.results is not None:
            self.logger.info("Loaded results from results.pkl")
            return
        self.results = []
        
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
            similarity = self._calculate_similarity(response, true_response)
            self.results.append((similarity, is_member))

            # 记录到表格中
            self.logger.new_row("gap-attack_results")
            self.logger.add("Input", attack_sample["input"])
            self.logger.add("True Response", true_response)
            self.logger.add("Generated", response)
            self.logger.add("Similarity", similarity)
            self.logger.add("Is member", is_member)
            self.logger.info(final_prompt)

            self.logger.info("-" * 50)
        
        self.logger.save()
        self.logger.save_data(self.results, "results.pkl")

    def evaluate(self) -> Dict[str, float]:
        # 将结果转换为DataFrame
        results_df = pd.DataFrame(self.results, columns=['similarity', 'ground_truth'])
        all_metrics = []
        all_roc_data = []  # 存储所有折的ROC数据
        
        # 进行n-fold交叉验证
        for fold in range(self.num_cross_validation):
            # 划分训练集和测试集
            train_df, test_df = train_test_split(
                results_df,
                train_size=self.train_attack,
                random_state=fold,
                shuffle=True
            )
            
            # 在训练集上找到最优阈值
            best_threshold, best_accuracy = EvaluationMetrics.get_best_threshold(
                train_df['ground_truth'], train_df['similarity']
            )
            
            # 在测试集上评估
            test_predictions = test_df['similarity'] >= best_threshold
            test_accuracy = np.mean(test_predictions == test_df['ground_truth'])
            
            # 计算ROC和AUC
            fpr, tpr, roc_auc = EvaluationMetrics.calculate_roc_auc(
                test_df['ground_truth'], test_df['similarity']
            )
            
            all_metrics.append({
                'accuracy': test_accuracy,
                'auc': roc_auc,
                'threshold': best_threshold
            })
            
            # 保存ROC数据
            all_roc_data.append({
                'fold': fold,
                'fpr': fpr.tolist(),  # 转换为list以便JSON序列化
                'tpr': tpr.tolist()
            })
        
        # 计算accuracy的均值和方差
        accuracies = [m['accuracy'] for m in all_metrics]
        best_thresholds = [m['threshold'] for m in all_metrics]
        final_metrics = {
            'avg_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'best_thresholds': best_thresholds,
            'roc_data': all_roc_data,  # 添加ROC数据到最终结果
        }
        
        self.logger.save_json('metrics.json', final_metrics)
        self.logger.info(f"Metrics: {final_metrics}")
        return final_metrics