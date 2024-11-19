import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

demos = 5
start = 1
end = demos+1

def load_data(base_path):
    data = []
    for i in range(start, end):  # selected_attack_sample from 1 to 6
        path = os.path.join(base_path, f"exp_num_demonstrations_{demos}_selected_attack_sample_{i}", "data", "scores.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                scores = json.load(f)
                data.extend([(i, score) for score in scores])
    return pd.DataFrame(data, columns=['selected_attack_sample', 'score'])

def plot_guitar(data):
    plt.figure(figsize=(12, 8))
    
    # Create the guitar plot
    sns.violinplot(x='selected_attack_sample', y='score', data=data, 
                   inner='box', cut=0, scale='area', width=0.9)
    
    # Add individual points
    sns.stripplot(x='selected_attack_sample', y='score', data=data, 
                  color='black', alpha=0.4, size=3, jitter=True)
    
    plt.title('Similarity Distribution by Selected Attack Sample (Guitar Plot)', fontsize=16)
    plt.xlabel('Selected Attack Sample')
    plt.ylabel('Similarity')
    
    # Adjust x-axis ticks
    plt.xticks(range(end-start), range(start, end))
    
    # Set y-axis limits
    plt.ylim(0, 1)
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('attack_sample_score_distribution_guitar.png', dpi=300)
    plt.close()

# 假设您的输出文件夹路径
base_path = 'cache'

# 加载数据
data = load_data(base_path)

# 绘制吉他图
plot_guitar(data)

print("吉他图已保存为 'attack_sample_score_distribution_guitar.png'")