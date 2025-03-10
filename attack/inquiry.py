from .common import *

from . import ICLAttackStrategy

class InquiryAttack(ICLAttackStrategy):
    def __init__(self, attack_config):
        super().__init__(attack_config)
        self.num_cross_validation = attack_config.get('cross_validation', 1)
        self.inquiry_template = attack_config.get('inquiry_template', "Have you seen this sentence before: {sample}?")
        self.positive_keywords = attack_config.get('positive_keywords', ["yes", "seen", "encountered", "familiar"])
        self.negative_keywords = attack_config.get('negative_keywords', ["no", "not seen", "unfamiliar"])

    def construct_inquiry(self, sample):
        return self.inquiry_template.format(input=sample)

    def is_member_by_response(self, response):
        words = [self.remove_punctuation(word.lower()) for word in self.remove_punctuation(response).split()]

        if len(words) == 0:
            return None

        # Check negative keywords
        if any(word in self.negative_keywords for word in words):
            return False
        if "have not seen" in response.lower() or "haven't seen" in response.lower():
            return False
        
        # Check positive keywords
        if any(word in self.positive_keywords for word in words):
            return True
        if "have seen" in response.lower() or "have encountered" in response.lower():
            return True
        
        if words[0].startswith("1"):
            return True
        elif words[0].startswith("0"):
            return False

        # Model did not provide valid information
        return None

    @ICLAttackStrategy.cache_results
    def attack(self, model):
        # Check if results.pkl already exists
        self.results = self.logger.load_data("results.pkl")
        if self.results is not None:
            self.logger.info("Loaded results from results.pkl")
            return
        self.results = []
        
        self.logger.new_table("inquiry-attack_results")
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
            
            # Record to table
            self.logger.new_row("inquiry-attack_results")
            self.logger.add("Input", attack_sample["input"])
            self.logger.add("Response", response)
            self.logger.add("Prediction", pred_member)
            self.logger.add("Is member", is_member)

            self.logger.info("-" * 50)
        
        self.logger.save()
        # Save results to file
        self.logger.save_data(self.results, "results.pkl")

    def evaluate(self) -> Dict[str, float]:
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results, columns=['prediction', 'ground_truth'])
        all_metrics = []
        all_roc_data = []  # Store ROC data for all folds
        
        # Calculate size for each fold
        fold_size = len(results_df) // self.num_cross_validation
        
        # Divide data into n parts
        for fold in range(self.num_cross_validation):
            # Use slicing to get data
            start_idx = fold * fold_size
            end_idx = (start_idx + fold_size) if fold < self.num_cross_validation - 1 else len(results_df)
            test_df = results_df.iloc[start_idx:end_idx]
            
            # Calculate accuracy directly
            test_accuracy = np.mean(test_df['prediction'] == test_df['ground_truth'])
            
            # Calculate ROC and AUC
            fpr, tpr, roc_auc = EvaluationMetrics.calculate_roc_auc(
                test_df['ground_truth'], test_df['prediction']
            )
            
            all_metrics.append({
                'accuracy': test_accuracy,
                'auc': roc_auc
            })
            
            # Save ROC data
            all_roc_data.append({
                'fold': fold,
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            })
        
        # Calculate mean and standard deviation of accuracy
        accuracies = [m['accuracy'] for m in all_metrics]
        final_metrics = {
            'avg_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'roc_data': all_roc_data
        }
        
        self.logger.save_json('metrics.json', final_metrics)
        self.logger.info(f"Metrics: {final_metrics}")
        return final_metrics