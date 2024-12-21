from .common import *

from attack import ICLAttackStrategy
from string_utils import StringHelper

from llm.query import ModelInterface

class RepeatAttack(ICLAttackStrategy):
    def __init__(self, attack_config):
        super().__init__(attack_config)
        self.num_words = attack_config.get('num_words', 3)
        self.similarity_threshold = attack_config.get('similarity_threshold', 0.8)
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
        self.logger.new_table("repeat-attack_results")

        train_data = self.data_loader.train()
        test_data = self.data_loader.test()
        data_list = train_data + test_data
        train_length = len(train_data)
        
        for i, (icl_samples, attack_sample, is_member) in enumerate(tqdm(data_list)):
            is_train = i < train_length
            icl_prompt = self.generate_icl_prompt(icl_samples, is_train=is_train)
            
            # Take num_words as the half length of the sentence if num_words is 0
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
            pred_member = similarity >= self.similarity_threshold
            
            self.results.append((pred_member, is_member, similarity))
            
            # 添加日志输出:")
            self.logger.new_row("repeat-attack_results")
            self.logger.add("Original", attack_sample["input"])
            self.logger.add("Expected", f"{latter} <or> {former} {latter if latter != '<empty>' else ''}")
            self.logger.add("Generated", f"{former_gen} {latter_gen if latter_gen != '<empty>' else ''}")
            self.logger.add("Similarity", similarity)
            self.logger.add("Is member", is_member)
            self.logger.add("Predicted member", pred_member)
            print("-" * 50)
        
        self.logger.save()

    def evaluate(self):
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