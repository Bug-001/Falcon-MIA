from .common import *

from attack import ICLAttackStrategy
from string_utils import StringHelper

from llm.query import ModelInterface

class RepeatAttack(ICLAttackStrategy):
    def __init__(self, attack_config):
        super().__init__(attack_config)
        self.num_cross_validation = attack_config.get('cross_validation', 1)
        self.num_words = attack_config.get('num_words', 3)
        self.shelper = StringHelper()

    def truncate_sample(self, sample, num_words):
        sample = self.remove_punctuation(sample)
        text_list = sample.split()
        if num_words > len(text_list):
            return sample, '<empty>'
        elif num_words == len(text_list):
            return ' '.join(text_list[:num_words]), '<empty>'
        else:
            return ' '.join(text_list[:num_words]), ' '.join(text_list[num_words:])

    def calculate_similarity(self, original, generated):
        orig_embedding = self.shelper.preprocess_text(original, 'semantic')
        gen_embedding = self.shelper.preprocess_text(generated, 'semantic')
        return self.shelper.semantic_cosine_similarity(orig_embedding, gen_embedding)

    @ICLAttackStrategy.cache_results
    def attack(self, model: 'ModelInterface'):
        # 查找results.pkl是否已经存在
        self.results = self.logger.load_data("results.pkl")
        if self.results is not None:
            self.logger.info("Loaded results from results.pkl")
            return
        self.results = []
        
        self.logger.new_table("repeat-attack_results")
        data_loader = self.data_loader.train() + self.data_loader.test()
        
        for icl_samples, attack_sample, is_member in tqdm(data_loader):
            icl_prompt = self.generate_icl_prompt(icl_samples)
            
            num_words = self.num_words if self.num_words > 0 else (len(attack_sample["input"]) // 2)
            former, latter = self.truncate_sample(attack_sample["input"], num_words)
            all_prompt = icl_prompt + [{
                "role": "user",
                "content": self.attack_config.get('repeat_template', "Complete the following sentence: {input}").format(input=former)
            }]
            generated_text = model.query(all_prompt, "Repeat Attack")[0]
            
            # LLM only generates the latter part
            similarity_1 = self.calculate_similarity(generated_text, latter)
            # LLM generates the whole sentence
            former_gen, latter_gen = self.truncate_sample(generated_text, num_words)
            similarity_2 = self.calculate_similarity(latter_gen, latter)
            similarity = max(similarity_1, similarity_2)
            
            self.results.append((similarity, is_member))
            
            # 记录到表格中
            self.logger.new_row("repeat-attack_results")
            self.logger.add("Original", attack_sample["input"])
            self.logger.add("Expected", f"{latter} <or> {former} {latter if latter != '<empty>' else ''}")
            self.logger.add("Generated", f"{former_gen} {latter_gen if latter_gen != '<empty>' else ''}")
            self.logger.add("Similarity", similarity)
            self.logger.add("Is member", is_member)
            
            self.logger.info("-" * 50)
        
        self.logger.save()
        self.logger.save_data(self.results, "results.pkl")

    def evaluate(self):
        # 将结果转换为DataFrame
        results_df = pd.DataFrame(self.results, columns=['similarity', 'ground_truth'])
        all_metrics = []
        all_roc_data = []  # 存储所有折的ROC数据
        
        # 进行n-fold交叉验证
        for fold in range(self.num_cross_validation):
            # 划分训练集和测试集
            train_df, test_df = train_test_split(
                results_df,
                train_size=self.train_attack,  # 使用和gap.py相同的训练集大小
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
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            })
        
        # 计算accuracy的均值和方差
        accuracies = [m['accuracy'] for m in all_metrics]
        best_thresholds = [m['threshold'] for m in all_metrics]
        final_metrics = {
            'avg_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'best_thresholds': best_thresholds,
            'roc_data': all_roc_data
        }
        
        self.logger.save_json('metrics.json', final_metrics)
        self.logger.info(f"Metrics: {final_metrics}")
        return final_metrics