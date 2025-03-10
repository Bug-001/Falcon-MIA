from .common import *

from attack import ICLAttackStrategy
from string_utils import ObfuscationTechniques, StringHelper
from attack.mitigation import create_privacy_mitigation

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from transformers import Trainer, TrainingArguments

from captum.attr import (
    IntegratedGradients,
    DeepLift,
    FeatureAblation,
    Occlusion
)
import seaborn as sns
from itertools import product

@dataclass
class ObfuscationDataCollator:
    def __call__(self, features: List[Tuple[Dict, bool]]) -> Dict[str, torch.Tensor]:
        # Separate features and labels
        features_list = []
        labels = []
        for data_point, label in features:
            # Get all unique similarity types
            similarity_types = sorted(list(set(
                sim_type for level_data in data_point.values() 
                for sim_type in level_data.keys()
            )))
            
            # Build feature vector
            feature_vector = []
            for level, sim_dict in sorted(data_point.items()):
                feature_vector.append(level)  # Add level value
                feature_vector.extend(sim_dict.get(sim_type, 0.0) for sim_type in similarity_types)
            
            features_list.append(feature_vector)
            labels.append(float(label))

        return {
            'features': torch.tensor(features_list, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }

class ObfuscationModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32, 16]):
        super().__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Dropout(0.3))
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.Dropout(0.3))
            
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, features, labels=None):
        logits = self.model(features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

class HFModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        return self.model(input)['logits']

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (logits > 0.5).astype(np.int64)
    
    accuracy = accuracy_score(labels, predictions)

    return {
        'accuracy': accuracy,
    }

class ObfuscationAttack(ICLAttackStrategy):
    def __init__(self, attack_config: Dict[str, Any]):
        super().__init__(attack_config)
        self.obfuscator = ObfuscationTechniques(attack_config)
        self.num_obfuscation_levels = attack_config.get('num_obfuscation_levels', 5)
        self.max_obfuscation_level = attack_config.get('max_obfuscation_level', 1)
        self.attack_template = attack_config.get('obsfucation_attack_template', "Classify the following text: {sample}")

        self.shelper = StringHelper()
        self.num_similarities = attack_config.get('num_similarities', 5)
        
        self.num_cross_validation = attack_config.get('cross_validation', 1)

        # Change single classifier to a list of classifiers
        self.classifiers = []
        for _ in range(self.num_cross_validation):
            classifier = ObfuscationModel(
                self.num_obfuscation_levels * (self.num_similarities + 1), 
                hidden_sizes=[128, 64, 32, 16]
            )
            classifier.to("cuda")
            self.classifiers.append(classifier)

        self.detector_dir = Path(self.logger.output_dir)
        # Training configs
        self.training_args = TrainingArguments(
            output_dir=self.detector_dir,
            num_train_epochs=attack_config.get('num_epochs', 100),
            per_device_train_batch_size=attack_config.get('batch_size', 8),
            per_device_eval_batch_size=attack_config.get('batch_size', 32),
            learning_rate=attack_config.get('learning_rate', 5e-5),
            weight_decay=attack_config.get('weight_decay', 0.01),
            logging_dir=self.detector_dir/"logs",
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            eval_steps=50,
            save_steps=50,
            save_total_limit=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to='none', # XXX: Disable wandb temporarily
        )

        # Add data storage lists
        self.train_data_list = [None] * self.num_cross_validation
        self.test_data_list = [None] * self.num_cross_validation

    def _attack(self, model, query_hooks=None):
        # Create main table and detailed record table
        results_table = "dataset_overview"
        details_table = "level_details"

        similarities_data = []
        
        self.logger.new_table(results_table)
        self.logger.new_table(details_table)

        train_length = self.train_attack
        data_loader = self.data_loader.train() + self.data_loader.test()
        for i, (icl_samples, attack_sample, is_member) in enumerate(tqdm(data_loader)):
            # Create new row for main table
            main_row = self.logger.new_row(results_table)

            # Determine if it's training data based on index
            is_train = i < train_length
            icl_prompt = self.generate_icl_prompt(icl_samples, is_train=is_train)
            
            # Select text to obfuscate
            input_text = attack_sample["input"].split()
            output_text = attack_sample["output"].split()
            if self.attack_config.get('selected_obf_text', 'all') == 'input':
                original_text = input_text
            elif self.attack_config.get('selected_obf_text', 'all') == 'output':
                original_text = output_text
            elif self.attack_config.get('selected_obf_text', 'all') == 'longest':
                original_text = input_text if len(input_text) > len(output_text) else output_text
            else: # 'all'
                original_text = input_text + output_text
            original_text = " ".join(original_text)
            
            # Record and print basic information
            self.logger.add("sample_id", main_row)
            self.logger.add("type", "train" if i < train_length else "test")
            self.logger.add("Original", original_text)
            self.logger.add("Membership", is_member)
            
            all_level_similarities = dict()
            # Process each obfuscation level
            for level in np.linspace(0.6, self.max_obfuscation_level, self.num_obfuscation_levels):
                level_row = self.logger.new_row(details_table)
                
                obfuscated_text = self.obfuscator.obfuscate(original_text, level=level)
                query_prompt = icl_prompt + [{
                    "role": "user",
                    "content": self.attack_template.format(input=obfuscated_text)
                }]
                
                # Apply privacy hooks if provided
                query_return = model.query(
                    query_prompt, 
                    "Obfuscation Attack",
                    query_hook_dict=query_hooks
                )
                response = query_return[0]
                similarities_first_line = self.shelper.calculate_overall_similarity_dict(original_text, response.split('\n')[0])
                similarities_all_lines = self.shelper.calculate_overall_similarity_dict(original_text, response)
                # Take maximum of each metric
                similarities = {k: max(v, similarities_first_line[k]) for k, v in similarities_all_lines.items()}
                
                # Record detailed level information
                self.logger.add("sample_id", main_row)
                self.logger.add("level", level)
                self.logger.add("obfuscated_text", obfuscated_text)
                self.logger.add("response", response)
                self.logger.add("similarities", similarities)
                
                all_level_similarities[level] = similarities

            similarities_data.append((all_level_similarities, is_member))

        return similarities_data

    def train_once(self, x: int):
        # Set save path
        model_dir = Path(f"obf-mlp-{x}")
        Path(self.detector_dir/model_dir).mkdir(parents=True, exist_ok=True)
        classfier_path = model_dir/"model.safetensors"
        
        # Data file path
        train_data_path = model_dir/"train_data.pkl"
        test_data_path = model_dir/"test_data.pkl"
        
        if not os.path.exists(self.detector_dir/classfier_path):
            # Split data using train_test_split
            train_data, test_data = train_test_split(
                self.similarities_data,
                train_size=self.train_attack,
                random_state=x,
                shuffle=True
            )
            
            # Save training and test sets
            self.logger.save_data(train_data, train_data_path)
            self.logger.save_data(test_data, test_data_path)
            
            # Save to instance variables
            self.train_data_list[x] = train_data
            self.test_data_list[x] = test_data
            
            # Update training_args output path
            self.training_args.output_dir = self.detector_dir/model_dir
            self.training_args.logging_dir = self.detector_dir/model_dir/"logs"

            trainer = Trainer(
                model=self.classifiers[x],  # Use corresponding classifier
                args=self.training_args,
                data_collator=ObfuscationDataCollator(),
                train_dataset=train_data,
                eval_dataset=test_data,
                compute_metrics=compute_metrics,
            )
            
            self.logger.info(f"Starting model-{x} training...")
            trainer.train()
            self.logger.info(f"Model-{x} training completed.")
            trainer.save_model(self.detector_dir/model_dir)
        else:
            self.logger.info(f"Model-{x} already trained. Loading the model...")
            from safetensors.torch import load_file
            self.classifiers[x].load_state_dict(load_file(self.detector_dir/classfier_path))
            train_data = self.logger.load_data(train_data_path)
            test_data = self.logger.load_data(test_data_path)
            if train_data is None or test_data is None:
                raise ValueError(f"Cannot find saved datasets for model-{x}")
            self.train_data_list[x] = train_data
            self.test_data_list[x] = test_data

    def test_once(self, x: int):
        self.classifiers[x].eval()
        test_data = self.test_data_list[x]
        
        # Process test_data through collator
        test_features = ObfuscationDataCollator()(test_data)
        
        predictions_values = []
        # Record prediction results for each sample
        for i, (feature_vector, is_member) in enumerate(zip(test_features['features'], test_features['labels'])):
            with torch.no_grad():
                pred = self.classifiers[x](feature_vector.to('cuda').unsqueeze(0))['logits'][0].to('cpu').item()
            predictions_values.append(pred)
            
            # Get original text and response
            original_text = test_data[i][0]
            all_level_similarities = test_data[i][0]
            responses = [f"Level {level}: {sims}" for level, sims in all_level_similarities.items()]
            is_member = bool(is_member)
            
            # Record prediction results, add model_id parameter
            self.__add_evaluation_record_to_table(
                "predictions", 
                i,
                original_text, 
                responses, 
                is_member, 
                all_level_similarities, 
                pred,
                x  # Pass model_id
            )
            
            # If prediction is wrong, record to mispredict table
            if (pred > 0.5) != is_member:
                self.__add_evaluation_record_to_table(
                    "mispredicted_results", 
                    i,
                    original_text, 
                    responses, 
                    is_member, 
                    all_level_similarities, 
                    pred,
                    x  # Pass model_id
                )

        # Calculate metrics
        predictions = list(map(lambda x: x > 0.5, predictions_values))
        ground_truth = test_features['labels'].tolist()
        
        metrics = EvaluationMetrics.calculate_advantage(predictions, ground_truth)
        
        # Calculate ROC curve and AUC
        fpr, tpr, roc_auc = EvaluationMetrics.calculate_roc_auc(ground_truth, predictions_values)
        metrics['auc'] = roc_auc
        
        return metrics

    def evaluate(self):
        mispredict_table = "mispredicted_results"
        predictions_table = "predictions"
        metrics_table = "metrics"
        self.logger.new_table(mispredict_table)
        self.logger.new_table(predictions_table)
        self.logger.new_table(metrics_table)

        # Collect metrics from all runs
        all_metrics = []
        num_runs = self.attack_config.get('cross_validation', 1)
        for i in range(num_runs):
            metrics = self.test_once(i)
            all_metrics.append(metrics)
            
            # Add metrics from each run to the table
            for key, value in metrics.items():
                self.logger.add(f"run_{i}_{key}", value, metrics_table)

        # Calculate mean and standard deviation
        avg_metrics = {}
        std_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
            std_metrics[key] = np.std(values)
            
            # Add to table
            self.logger.add(f"avg_{key}", avg_metrics[key], metrics_table)
            self.logger.add(f"std_{key}", std_metrics[key], metrics_table)

        # Use feature techniques to analyze the impact of obfuscation level and similarity
        # XXX: Very confusing code...
        self.analyze_feature_importance(
            HFModelWrapper(self.classifiers[0]), # Only analyze the first one
            ObfuscationDataCollator()(self.similarities_data)['features'].to('cuda'),
            list(self.similarities_data[0][0].keys()),
            ['level'] + sorted(list(self.similarities_data[0][0].values())[0].keys()),
        )

        # Save results
        self.logger.save()
        return avg_metrics

    def attack(self, model):
        self.similarities_data = self.logger.load_data("similarities_data")
        if self.similarities_data == None:
            if self.attack_config.get('attack_phase', 'all') == 'train-test':
                raise ValueError("Similarities data is not found. Please run the attack in all phase")
            
            # Set up privacy mitigation hooks if enabled in config
            privacy_hooks = create_privacy_mitigation(self.attack_config, model)
            
            if self.attack_config.get('sim_use_idf', False):
                self.idf = self.sdm.get_idf()
                self.shelper.set_idf_dict(self.idf)
            if self.attack_config.get('obf_use_idf', False):
                self.idf = self.sdm.get_idf()
                self.obfuscator.set_idf_dict(self.idf)
            
            # Apply privacy hooks during attack if enabled
            self.similarities_data = self._attack(
                model, 
                query_hooks=privacy_hooks,
            )
            self.logger.save_data(self.similarities_data, "similarities_data")
            self.logger.save()
        else:
            print("Loaded cached similarities data.")

        if self.attack_config.get('attack_phase', 'all') == 'request':
            raise KeyboardInterrupt

        # Train models
        for i in range(self.num_cross_validation):
            self.train_once(i)

    def __add_evaluation_record_to_table(self, table, sample_id, original_text, responses, is_member, similarities, prediction, model_id):
        row = self.logger.new_row(table)
        self.logger.add("sample_id", sample_id, table, show=False)
        self.logger.add("Original", original_text, table, show=False)
        self.logger.add("Responses", responses, table, show=False)
        self.logger.add("Membership", is_member, table, show=False)
        self.logger.add("Similarities", similarities, table, show=False)
        self.logger.add("Prediction", prediction, table, show=False)
        self.logger.add("model_id", model_id, table, show=False)

    def analyze_feature_importance(self, model, input_features, levels, similarity_types):
        # Ensure model is in evaluation mode
        model.eval()
        
        # 1. Integrated Gradients analysis
        ig = IntegratedGradients(model)
        ig_attr = ig.attribute(input_features, target=0)  # target=0 because output is 1-dimensional
        
        # 2. DeepLift analysis
        dl = DeepLift(model)
        dl_attr = dl.attribute(input_features, target=0)
        
        # 3. Feature Ablation analysis
        ablator = FeatureAblation(model)
        abl_attr = ablator.attribute(input_features, target=0)
        
        # Calculate importance score for each feature (mean of absolute values)
        importance_scores = {
            'integrated_gradients': torch.abs(ig_attr).mean(dim=0),
            'deeplift': torch.abs(dl_attr).mean(dim=0),
            'ablation': torch.abs(abl_attr).mean(dim=0)
        }

        # Create figure
        plt.figure(figsize=(20, 5))
        
        # XXX: Temporarily truncate the first column of level
        similarity_types = similarity_types[1:]

        # Create a subplot for each method
        for idx, (method_name, scores) in enumerate(importance_scores.items(), 1):
            # Create heatmap data matrix
            heatmap_data = np.zeros((len(levels), len(similarity_types)))
            
            # Fill data
            for feature_idx, (level, sim) in enumerate(product(levels, similarity_types)):
                i = levels.index(level)
                j = similarity_types.index(sim)
                heatmap_data[i, j] = scores[feature_idx].item()

            # Create subplot
            plt.subplot(1, 3, idx)
            sns.heatmap(
                heatmap_data,
                xticklabels=similarity_types,
                yticklabels=levels,
                annot=False,
                fmt='.3f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Importance Score'},
            )
            
            plt.title(f'{method_name.replace("_", " ").title()}')
            plt.xlabel('Similarity')
            plt.ylabel('Level')
        
        plt.tight_layout()
        self.logger.savefig('feature_importance_heatmap.png')