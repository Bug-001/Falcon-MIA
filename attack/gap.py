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
        # Store optimal thresholds and results for each fold
        self.thresholds = []
        self.fold_results = []
        
    def _calculate_similarity(self, response: str, true_output: str) -> float:
        """Helper method: Calculate similarity"""
        response_embedding = self.shelper.preprocess_text(response, 'semantic')
        true_embedding = self.shelper.preprocess_text(true_output, 'semantic')
        return self.shelper.semantic_cosine_similarity(response_embedding, true_embedding)

    @ICLAttackStrategy.cache_results
    def attack(self, model: 'ModelInterface'):
        # Check if results.pkl already exists
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

            # Calculate similarity and record results
            similarity = self._calculate_similarity(response, true_response)
            self.results.append((similarity, is_member))

            # Record to table
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
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results, columns=['similarity', 'ground_truth'])
        all_metrics = []
        all_roc_data = []  # Store ROC data for all folds
        
        # Perform n-fold cross validation
        for fold in range(self.num_cross_validation):
            # Split into training and test sets
            train_df, test_df = train_test_split(
                results_df,
                train_size=self.train_attack,
                random_state=fold,
                shuffle=True
            )
            
            # Find optimal threshold on training set
            best_threshold, best_accuracy = EvaluationMetrics.get_best_threshold(
                train_df['ground_truth'], train_df['similarity']
            )
            
            # Evaluate on test set
            test_predictions = test_df['similarity'] >= best_threshold
            test_accuracy = np.mean(test_predictions == test_df['ground_truth'])
            
            # Calculate ROC and AUC
            fpr, tpr, roc_auc = EvaluationMetrics.calculate_roc_auc(
                test_df['ground_truth'], test_df['similarity']
            )
            
            all_metrics.append({
                'accuracy': test_accuracy,
                'auc': roc_auc,
                'threshold': best_threshold
            })
            
            # Save ROC data
            all_roc_data.append({
                'fold': fold,
                'fpr': fpr.tolist(),  # Convert to list for JSON serialization
                'tpr': tpr.tolist()
            })
        
        # Calculate mean and standard deviation of accuracy
        accuracies = [m['accuracy'] for m in all_metrics]
        best_thresholds = [m['threshold'] for m in all_metrics]
        final_metrics = {
            'avg_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'best_thresholds': best_thresholds,
            'roc_data': all_roc_data,  # Add ROC data to final results
        }
        
        self.logger.save_json('metrics.json', final_metrics)
        self.logger.info(f"Metrics: {final_metrics}")
        return final_metrics