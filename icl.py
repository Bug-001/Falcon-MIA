# icl_attack.py

import yaml
import argparse
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional

from llm.tools.utils import get_logger
from llm.query import ModelInterface
from attack import ICLAttackStrategy

logger = get_logger("ICL Attack", "info")
# class BaseTrainer(ABC):
#     def __init__(self, config: Dict[str, Any], logger):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.epochs = config.get('epochs', 1000)
#         self.log_interval = self.epochs // 10
#         self.batch_size = config.get('batch_size', 32)
#         self.learning_rate = config.get('learning_rate', 0.001)
#         self.save_plots = config.get('save_plots', True)
#         self.logger = logger
#         self.save_plots = config.get('save_plots', True)
#         self.early_stopping_patience = config.get('early_stopping_patience', 30)
#         self.early_stopping_delta = config.get('early_stopping_delta', 1e-4)
        
#     @abstractmethod
#     def get_model(self) -> nn.Module:
#         """Return the model to be trained."""
#         pass
    
#     @abstractmethod
#     def get_criterion(self) -> nn.Module:
#         """Return the loss criterion."""
#         pass
    
#     def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
#         """Return the optimizer. Can be overridden if needed."""
#         return optim.Adam(model.parameters(), lr=self.learning_rate)
    
#     def prepare_data(self, data, labels) -> DataLoader:
#         """Convert data to DataLoader. Can be overridden for custom data preparation."""
#         dataset = TensorDataset(
#             torch.tensor(data, dtype=torch.float32).to(self.device),
#             torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(self.device)
#         )
#         return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
    
#     def evaluate_epoch(self, 
#                       model: nn.Module, 
#                       dataloader: DataLoader, 
#                       criterion: nn.Module) -> Tuple[float, float]:
#         """Evaluate model on the given dataloader."""
#         model.eval()
#         total_loss = 0.0
#         correct_preds = 0
#         total_samples = 0
        
#         with torch.no_grad():
#             for data, labels in dataloader:
#                 outputs = model(data)
#                 loss = criterion(outputs, labels)
#                 total_loss += loss.item() * len(data)
                
#                 predictions = (outputs > 0.5).float()
#                 correct_preds += (predictions == labels).sum().item()
#                 total_samples += len(data)
        
#         avg_loss = total_loss / total_samples
#         accuracy = correct_preds / total_samples
#         return avg_loss, accuracy

#     def early_stopping_check(self, 
#                            current_loss: float, 
#                            best_loss: float, 
#                            patience_counter: int) -> Tuple[float, int, bool]:
#         """Check if training should be stopped early."""
#         if current_loss < best_loss - self.early_stopping_delta:
#             best_loss = current_loss
#             patience_counter = 0
#         else:
#             patience_counter += 1
            
#         should_stop = patience_counter >= self.early_stopping_patience
#         return best_loss, patience_counter, should_stop

#     def train(self, 
#              train_data: torch.Tensor, 
#              train_labels: torch.Tensor,
#              test_data: Optional[torch.Tensor] = None,
#              test_labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
#         """
#         Train the model with both training and test data.
        
#         Returns:
#             Dict containing trained model and training history
#         """
#         model = self.get_model().to(self.device)
#         criterion = self.get_criterion()
#         optimizer = self.get_optimizer(model)
        
#         # Prepare data loaders
#         train_loader = self.prepare_data(train_data, train_labels)
#         test_loader = None if test_data is None else self.prepare_data(test_data, test_labels)
        
#         # Initialize tracking variables
#         train_losses = []
#         train_accuracies = []
#         test_losses = []
#         test_accuracies = []
#         best_loss = float('inf')
#         best_model_state = None
#         patience_counter = 0
        
#         progress_bar = tqdm(range(self.epochs), desc="Training")
#         for epoch in progress_bar:
#             # Training phase
#             model.train()
#             epoch_loss = 0.0
#             correct_preds = 0
#             total_samples = 0
            
#             for batch_data, batch_labels in train_loader:
#                 optimizer.zero_grad()
#                 outputs = model(batch_data)
#                 loss = criterion(outputs, batch_labels)
#                 loss.backward()
#                 optimizer.step()
                
#                 epoch_loss += loss.item() * len(batch_data)
#                 predictions = (outputs > 0.5).float()
#                 correct_preds += (predictions == batch_labels).sum().item()
#                 total_samples += len(batch_data)
            
#             # Calculate training metrics
#             train_loss = epoch_loss / total_samples
#             train_accuracy = correct_preds / total_samples
#             train_losses.append(train_loss)
#             train_accuracies.append(train_accuracy)
            
#             should_stop = False

#             # Evaluation phase
#             if test_loader is not None:
#                 test_loss, test_accuracy = self.evaluate_epoch(model, test_loader, criterion)
#                 test_losses.append(test_loss)
#                 test_accuracies.append(test_accuracy)
                
#                 # Early stopping check
#                 best_loss, patience_counter, should_stop = self.early_stopping_check(
#                     test_loss, best_loss, patience_counter)
                
#                 if test_loss < best_loss:
#                     best_model_state = model.state_dict().copy()
                
#                 # Update progress bar
#                 progress_bar.set_postfix({
#                     'train_loss': f'{train_loss:.4f}',
#                     'train_acc': f'{train_accuracy:.4f}',
#                     'val_loss': f'{test_loss:.4f}',
#                     'val_acc': f'{test_accuracy:.4f}'
#                 })
#             else:
#                 progress_bar.set_postfix({
#                     'train_loss': f'{train_loss:.4f}',
#                     'train_acc': f'{train_accuracy:.4f}'
#                 })
            
#             # Logging
#             if (epoch + 1) % self.log_interval == 0:
#                 log_msg = f"Epoch [{epoch+1}/{self.epochs}], "
#                 log_msg += f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}"
#                 if test_loader is not None:
#                     log_msg += f", Val Loss: {test_loss:.4f}, Val Acc: {test_accuracy:.4f}"
#                 logger.info(log_msg)
            
#             if False:
#                 logger.info(f"Early stopping triggered at epoch {epoch+1}")
#                 break
        
#         # Load best model if available
#         if best_model_state is not None:
#             model.load_state_dict(best_model_state)
        
#         if self.save_plots:
#             self._plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies)
        
#         return {
#             'model': model,
#             'train_losses': train_losses,
#             'train_accuracies': train_accuracies,
#             'test_losses': test_losses,
#             'test_accuracies': test_accuracies,
#             'best_model_state': best_model_state
#         }
    
#     def predict(self, model: nn.Module, data) -> torch.Tensor:
#         """Make predictions using the trained model."""
#         model.eval()
#         with torch.no_grad():
#             if not isinstance(data, torch.Tensor):
#                 data = torch.tensor(data, dtype=torch.float32).to(self.device)
#             return model(data)
    
#     def _plot_training_curves(self, 
#                             train_losses: list, 
#                             test_losses: list,
#                             train_accuracies: list, 
#                             test_accuracies: list):
#         """Plot and save the training curves."""
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
#         # Plot losses
#         ax1.plot(train_losses, label='Train Loss')
#         if test_losses:
#             ax1.plot(test_losses, label='Val Loss')
#         ax1.set_title('Training and Val Losses')
#         ax1.set_xlabel('Epoch')
#         ax1.set_ylabel('Loss')
#         ax1.legend()
#         ax1.grid(True)
        
#         # Plot accuracies
#         ax2.plot(train_accuracies, label='Train Accuracy')
#         if test_accuracies:
#             ax2.plot(test_accuracies, label='Val Accuracy')
#         ax2.set_title('Training and Val Accuracies')
#         ax2.set_xlabel('Epoch')
#         ax2.set_ylabel('Accuracy')
#         ax2.legend()
#         ax2.grid(True)
        
#         plt.tight_layout()
#         self.logger.savefig('training_curves.png')
#         plt.close()
# # TODO: 仍未测试正确性

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(data_config, attack_config, query_config, name='unnamed_experiment'):
    if 'name' not in attack_config:
        attack_config['name'] = name
    attack_strategy = ICLAttackStrategy.create(attack_config)
    if attack_strategy is None:
        raise ValueError(f"Attack type {attack_config['type']} is not supported.")
    model = ModelInterface(query_config)
    attack_strategy.prepare(data_config)
    try:
        attack_strategy.attack(model)
    except KeyboardInterrupt:
        pass
    attack_strategy.evaluate()

if __name__ == "__main__":
    # Keep the original command-line argument parsing for backward compatibility
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='Path to the data config file', default="data.yaml")
    parser.add_argument('--attack', help='Path to the attack config file', default="attack_chat.yaml")
    parser.add_argument('--query', help='Path to the query config file', default="query.yaml")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help="Set the logging level")
    parser.add_argument("--output", help="Output directory for all results", default=None)
    args = parser.parse_args()

    data_config = load_yaml_config(args.data)
    attack_config = load_yaml_config(args.attack)
    query_config = load_yaml_config(args.query)
    main(data_config, attack_config, query_config)