class RepeatAttack(ICLAttackStrategy):
    def __init__(self, attack_config):
        super().__init__(attack_config)
        self.num_words = attack_config.get('num_words', 3)
        self.similarity_threshold = attack_config.get('similarity_threshold', 0.8)
        self.similarity_metric = attack_config.get('similarity_metric', 'cosine')
        self.encoder = None  # 延迟初始化编码器

    def initialize_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder_name = self.attack_config.get('encoder', 'paraphrase-MiniLM-L6-v2')
            self.encoder = SentenceTransformer(encoder_name)

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
        self.initialize_encoder()  # 确保编码器已初始化
        original_embedding = self.encoder.encode([original])[0]
        generated_embedding = self.encoder.encode([generated])[0]
        
        if self.similarity_metric == 'cosine':
            return 1 - cosine(original_embedding, generated_embedding)
        elif self.similarity_metric == 'euclidean':
            return 1 / (1 + euclidean(original_embedding, generated_embedding))
        elif self.similarity_metric == 'manhattan':
            return 1 / (1 + cityblock(original_embedding, generated_embedding))
        elif self.similarity_metric == 'dot_product':
            return np.dot(original_embedding, generated_embedding) / (np.linalg.norm(original_embedding) * np.linalg.norm(generated_embedding))
        elif self.similarity_metric == 'jaccard':
            # 对于Jaccard相似度，我们需要将向量转换为集合
            set1 = set(np.where(original_embedding > np.mean(original_embedding))[0])
            set2 = set(np.where(generated_embedding > np.mean(generated_embedding))[0])
            return len(set1.intersection(set2)) / len(set1.union(set2))
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

    @ICLAttackStrategy.cache_results
    def attack(self, model):
        for icl_samples, attack_sample, is_member in tqdm(self.data_loader.test()):
            icl_prompt = self.generate_icl_prompt(icl_samples)
            
            # Take num_words as the half length of the sentence if num_words is 0
            num_words = self.num_words if self.num_words > 0 else (len(attack_sample["input"]) // 2)
            former, latter = self.truncate_sample(attack_sample["input"], num_words)
            all_prompt = icl_prompt + [{
                "role": "user",
                "content": self.attack_config.get('repeat_template', "Complete the following sentence: {input}").format(input=former)
            }]
            generated_text = model.query(all_prompt, "Repeat Attack")[0]
            
            # LLM only generates latter
            similarity_1 = self.calculate_similarity(generated_text, latter)
            # LLM generates the whole sentence
            former_gen, latter_gen = self.truncate_sample(generated_text, num_words)
            similarity_2 = self.calculate_similarity(latter_gen, latter)
            similarity = max(similarity_1, similarity_2)
            pred_member = similarity >= self.similarity_threshold
            
            self.results.append((pred_member, is_member, similarity))
            
            # 添加日志输出:")
            logger.info(f"Original: {attack_sample['input']}")
            logger.info(f"Expected: {latter} <or> {former} {latter if latter != '<empty>' else ''}")
            logger.info(f"Generated: {former_gen} {latter_gen if latter_gen != '<empty>' else ''}")
            logger.info(f"Similarity: {similarity}")
            logger.info(f"Is member: {is_member}, Predicted member: {pred_member}")
            logger.info("-" * 50)

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
        if self.attack_config.get('plot_roc', False):
            EvaluationMetrics.plot_roc(fpr, tpr, roc_auc, f'roc_curve_{self.__class__.__name__}.png')
        if self.attack_config.get('plot_log_roc', False):
            EvaluationMetrics.plot_log_roc(log_fpr, log_tpr, log_auc, f'log_roc_curve_{self.__class__.__name__}.png')

        return metrics