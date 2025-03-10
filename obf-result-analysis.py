import os
from pathlib import Path
import re
from typing import Dict, List, Tuple, Any
import pandas as pd
import functools
import numpy as np
from typing import Callable, Any
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import math
from tqdm import tqdm

from utils import SDatasetManager
from string_utils import StringHelper

def file_cache(filename: str):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if os.path.exists(filename):
                print(f"[{func.__name__}] Loading from cache {filename}")
                return pd.read_json(filename)
            file_dir = Path(filename).parent
            os.makedirs(file_dir, exist_ok=True)
            result: pd.DataFrame = func(*args, **kwargs)
            result.to_json(filename)
            print(f"[{func.__name__}] Data cached in {filename}")
            return result
        return wrapper
    return decorator

def parse_folder_name(folder_name: str) -> Dict[str, str]:
    """Parse path in format task(xx)--dataset(xx)--num_demonstrations(xx)--technique(xx)"""
    pattern = r"task\((\w+)\)--dataset\((\w+)\)--num_demonstrations\((\d+)\)--technique\((\w+)\)"
    match = re.match(pattern, folder_name)
    if not match:
        return {}
    return {
        "task": match.group(1),
        "dataset": match.group(2),
        "num_demonstrations": int(match.group(3)),
        "technique": match.group(4)
    }

def get_folders(root_dir: str) -> list:
    root = Path(root_dir)
    return [f for f in root.iterdir() if f.is_dir()]

def load_predictions(folder_path: Path) -> List[Tuple[Dict, Dict]]:
    pred_path = folder_path / "predictions.json"
    if not pred_path.exists():
        return None
    with open(pred_path, 'r') as f:
        return pd.read_json(f)

def set_effective_length(data: pd.DataFrame):
    shelper = StringHelper()

    def calculate_effective_length(row):
        cleaned: List[str] = shelper.preprocess_text(row['Original'], mode='idf')
        idf_sum = sum(shelper.idf_dict[word] if word in shelper.idf_dict else 0 for word in cleaned)
        return idf_sum / shelper.max_idf

    for (dataset, task), group in tqdm(data.groupby(['dataset', 'task']), desc='Calculating effective length'):
        sdm = SDatasetManager(dataset, task)
        shelper.set_idf_dict(sdm.get_idf())
        data.loc[group.index, 'effective_length'] = group.apply(calculate_effective_length, axis=1)

@file_cache("cache/log/obf-result-analysis/predictions.json")
def load_all_predictions():
    root_dir = Path("cache/log")
    model_name = 'Meta-Llama-3-8B-Instruct'
    folders = get_folders(root_dir/model_name/'obf_technique_test')

    # Collect data
    pred_data = pd.DataFrame()
    for folder in folders:
        info = eval(folder.name)
            
        predictions = load_predictions(folder)
        if predictions is None:
            continue

        # Add attack info to predictions
        predictions['dataset'] = info[1]
        predictions['task'] = info[0]
        predictions['num_demonstrations'] = info[2]
        predictions['length'] = predictions['Original'].apply(str.split).apply(len)

        pred_data = pd.concat([pred_data, predictions], ignore_index=True)

    set_effective_length(pred_data)

    return pred_data

@file_cache("cache/log/obf-result-analysis/datasets.json")
def load_all_datasets():
    root_dir = Path("cache/log")
    model_name = 'Meta-Llama-3-8B-Instruct'
    folders = get_folders(root_dir/model_name/'obf_technique_test')

    # Collect data
    pred_data = pd.DataFrame()
    for folder in folders:
        info = eval(folder.name)
            
        pred_path = folder / "dataset_overview.json"
        if not pred_path.exists():
            continue
        with open(pred_path, 'r') as f:
            predictions = pd.read_json(f)

        # Add attack info to predictions
        predictions['dataset'] = info[1]
        predictions['task'] = info[0]
        predictions['num_demonstrations'] = info[2]
        predictions['length'] = predictions['Original'].apply(str.split).apply(len)

        pred_data = pd.concat([pred_data, predictions], ignore_index=True)

    set_effective_length(pred_data)

    return pred_data

def plot_length_bias_density(data, figsize=(15, 8), method='kde', bins=50, std_multiplier=1):
    """
    Create a density distribution plot for bias values across different lengths.
    
    Parameters:
    data: pandas DataFrame with 'length' and 'bias' columns
    figsize: tuple of figure dimensions
    method: 'kde' or 'hist' for different density estimation methods
    bins: number of bins for bias values
    std_multiplier: number of standard deviations to show (default=1)
    """
    plt.figure(figsize=figsize)
    
    # Get all unique length values
    lengths = sorted(data['length'].unique())
    
    # Calculate bias range
    bias_min, bias_max = data['bias'].min(), data['bias'].max()
    bias_range = np.linspace(bias_min, bias_max, bins)
    
    # Create density matrix
    density_matrix = np.zeros((len(bias_range)-1, len(lengths)))
    
    if method == 'kde':
        # Use KDE method to calculate density
        for i, length in enumerate(lengths):
            length_data = data[data['length'] == length]['bias']
            if len(length_data) > 1:
                kde = stats.gaussian_kde(length_data)
                density = kde(bias_range)
                density_matrix[:, i] = np.interp(bias_range[:-1], 
                                               bias_range, 
                                               density/density.max())
    else:
        # Use histogram method to calculate density
        for i, length in enumerate(lengths):
            length_data = data[data['length'] == length]['bias']
            if len(length_data) > 0:
                hist, _ = np.histogram(length_data, bins=bias_range, density=True)
                density_matrix[:, i] = hist / hist.max() if hist.max() > 0 else hist
    
    # Create custom color map (light to dark blue)
    colors = ['#ffffff', '#e6f3ff', '#b3d9ff', '#80bfff', '#4da6ff', '#1a8cff', '#0066cc']
    cmap = LinearSegmentedColormap.from_list('custom_blues', colors)
    
    # Draw density plot
    plt.imshow(density_matrix, aspect='auto', origin='lower', 
              extent=[min(lengths), max(lengths), bias_min, bias_max],
              cmap=cmap, interpolation='nearest')
    
    # Add color bar
    plt.colorbar(label='Normalized Density')
    
    # Calculate mean and standard deviation
    stats_df = data.groupby('length')['bias'].agg([
        'mean',
        'std'
    ]).reset_index()
    stats_df.columns = ['length', 'mean', 'std']
    
    # Draw mean line and standard deviation range
    plt.plot(stats_df['length'], stats_df['mean'], 'r-', 
            linewidth=2, label='Mean')
    
    # Draw ±n standard deviations range
    plt.plot(stats_df['length'], 
            stats_df['mean'] + std_multiplier * stats_df['std'], 
            'g--', linewidth=1.5, 
            label=f'Mean ± {std_multiplier} Std Dev')
    plt.plot(stats_df['length'], 
            stats_df['mean'] - std_multiplier * stats_df['std'], 
            'g--', linewidth=1.5)
    
    # Set chart style
    plt.xlabel('Length', fontsize=12)
    plt.ylabel('Bias', fontsize=12)
    plt.title('Density Distribution of Bias Across Length Values', fontsize=14)
    plt.legend()
    
    # Optimize x-axis ticks based on data range
    tick_step = max(1, len(lengths) // 10)
    plt.xticks(np.arange(min(lengths), max(lengths)+1, tick_step))
    plt.ylim(math.floor(bias_min), math.ceil(bias_max))
    
    return plt

def main():
    output_path = Path("cache/log/obf-result-analysis")

    pred_data: pd.DataFrame = load_all_predictions()
    datasets: pd.DataFrame = load_all_datasets()

    # Plot scatter between length and attack score
    pred_data['bias'] = pred_data['Prediction'] - pred_data['Membership'].astype(float)
    # plot_length_bias_density(pred_data[pred_data['Membership'] == False], method='hist')
    # plt.savefig(output_path/'length_vs_bias_nonmember.png')
    # plot_length_bias_density(pred_data[pred_data['Membership'] == True], method='hist')
    # plt.savefig(output_path/'length_vs_bias_member.png')

    sns.histplot(data=pred_data, x='effective_length', hue='dataset', element='step', stat='density', common_norm=False)
    plt.savefig(output_path/'effective_length_hist.png')

if __name__ == "__main__":
    main()
