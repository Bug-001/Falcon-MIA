from pathlib import Path
import torch
from itertools import product
from attack.obfuscation import ObfuscationModel, ObfuscationDataCollator
from safetensors.torch import load_file
import pickle
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define global base path
BASE_PATH = Path("cache/log")  # Modify this according to actual situation
TASK = Path("obf_technique_test/('judgment', 'lexglue', 3)")
SIMDATA = Path("similarities_data")
DETECTOR = Path("obf-mlp-0/model.safetensors")

target_models = [
    "Llama-3-8B",
    # "Qwen-0.5B",
    # "Qwen-1.5B",
    "Qwen-3B",
    "Qwen-7B",
    "Qwen-14B",
    "gpt3.5-turbo",
    "gpt-4o-mini",
    "Ministral-8B",
]

def load_data_and_model(model_name: str):
    """
    Load data and trained detector for a specific model
    """
    # Assume data and model are under BASE_PATH/model_name
    task_path = BASE_PATH / model_name / TASK
    
    # Load data
    with open(task_path / SIMDATA, "rb") as f:
        data = pickle.load(f)
    
    # Process data using ObfuscationDataCollator
    collator = ObfuscationDataCollator()
    processed_data = collator(data)
    
    # Calculate input feature dimension
    input_size = processed_data['features'].shape[1]
    
    # Initialize model
    model = ObfuscationModel(input_size=input_size)
    model.load_state_dict(load_file(task_path / DETECTOR))
    model.to("cuda")
    model.eval()
    
    return processed_data, model

def plot_accuracy_heatmap(results, save_path="accuracy_heatmap.png"):
    """
    Plot cross-test results as a heatmap
    
    Args:
        results: Dict[Tuple[str, str], Dict] - Cross-test results
        save_path: str - Path to save the image
    """
    # Create accuracy matrix
    accuracy_matrix = np.zeros((len(target_models), len(target_models)))
    
    # Fill accuracy matrix
    for i, model_A in enumerate(target_models):
        for j, model_B in enumerate(target_models):
            if (model_A, model_B) in results:
                accuracy_matrix[i, j] = results[(model_A, model_B)]['accuracy']
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Draw heatmap
    sns.heatmap(
        accuracy_matrix,
        xticklabels=target_models,
        yticklabels=target_models,
        annot=True,  # Show values
        fmt='.3f',   # Value format
        cmap='YlOrRd',  # Color map
        vmin=0.5,    # Minimum value
        vmax=1.0,    # Maximum value
        cbar_kws={'label': 'Accuracy'}  # Color bar label
    )
    
    # Set labels
    plt.xlabel('Attack LLM')
    plt.ylabel('Target LLM')
    plt.title('Cross-Model Membership Inference Accuracy')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def cross_model_test():
    # Store all model data and detectors
    data_dict = {}
    model_dict = {}
    
    # Load all data and models
    print("Loading data and models...")
    for model_name in target_models:
        try:
            data, model = load_data_and_model(model_name)
            data_dict[model_name] = data
            model_dict[model_name] = model
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue
    
    # Cross testing
    print("Starting cross-model testing...")
    results = {}
    
    # Iterate through all possible model pairs
    for model_A, model_B in tqdm(list(product(target_models, target_models))):
        if model_A not in data_dict or model_B not in model_dict:
            continue
        
        # Get model A's data
        data = data_dict[model_A]
        # Get model B's detector
        classifier = model_dict[model_B]
        
        # Make predictions
        with torch.no_grad():
            predictions = classifier(data['features'].to('cuda'))['logits']
            predictions = predictions.cpu().squeeze()
        
        # Calculate accuracy
        labels = data['labels']
        correct = ((predictions > 0.5).float() == labels).float().mean().item()
        
        results[(model_A, model_B)] = {
            'accuracy': correct,
            # 'predictions': predictions.numpy(),
            # 'labels': labels.numpy()
        }
    
    # Draw heatmap
    plot_accuracy_heatmap(results)
    
    return results

if __name__ == "__main__":
    results = cross_model_test()
    # Format and print results
    models = sorted(list(set([m for pair in results.keys() for m in pair])))
    print("\nAccuracy Matrix:")
    print("Data\\Model", end="\t")
    for model in models:
        print(f"{model}", end="\t")
    print()
    
    for model_A in models:
        print(f"{model_A}", end="\t")
        for model_B in models:
            acc = results.get((model_A, model_B), {}).get('accuracy', 0.0)
            print(f"{acc:.3f}", end="\t")
        print()