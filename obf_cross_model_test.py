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

# 定义全局基础路径
BASE_PATH = Path("cache/log")  # 这里需要根据实际情况修改
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
    加载特定模型的数据和训练好的判别器
    """
    # 假设数据和模型都在BASE_PATH/model_name下
    task_path = BASE_PATH / model_name / TASK
    
    # 加载数据
    with open(task_path / SIMDATA, "rb") as f:
        data = pickle.load(f)
    
    # 使用ObfuscationDataCollator处理数据
    collator = ObfuscationDataCollator()
    processed_data = collator(data)
    
    # 计算输入特征维度
    input_size = processed_data['features'].shape[1]
    
    # 初始化模型
    model = ObfuscationModel(input_size=input_size)
    model.load_state_dict(load_file(task_path / DETECTOR))
    model.to("cuda")
    model.eval()
    
    return processed_data, model

def plot_accuracy_heatmap(results, save_path="accuracy_heatmap.png"):
    """
    将交叉测试结果绘制成热力图
    
    Args:
        results: Dict[Tuple[str, str], Dict] - 交叉测试的结果
        save_path: str - 保存图片的路径
    """
    # 创建准确率矩阵
    accuracy_matrix = np.zeros((len(target_models), len(target_models)))
    
    # 填充准确率矩阵
    for i, model_A in enumerate(target_models):
        for j, model_B in enumerate(target_models):
            if (model_A, model_B) in results:
                accuracy_matrix[i, j] = results[(model_A, model_B)]['accuracy']
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制热力图
    sns.heatmap(
        accuracy_matrix,
        xticklabels=target_models,
        yticklabels=target_models,
        annot=True,  # 显示数值
        fmt='.3f',   # 数值格式
        cmap='YlOrRd',  # 色彩映射
        vmin=0.5,    # 最小值
        vmax=1.0,    # 最大值
        cbar_kws={'label': 'Accuracy'}  # 颜色条标签
    )
    
    # 设置标签
    plt.xlabel('Attack LLM')
    plt.ylabel('Target LLM')
    plt.title('Cross-Model Membership Inference Accuracy')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def cross_model_test():
    # 存储所有模型的数据和判别器
    data_dict = {}
    model_dict = {}
    
    # 加载所有数据和模型
    print("Loading data and models...")
    for model_name in target_models:
        try:
            data, model = load_data_and_model(model_name)
            data_dict[model_name] = data
            model_dict[model_name] = model
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue
    
    # 交叉测试
    print("Starting cross-model testing...")
    results = {}
    
    # 遍历所有可能的模型对
    for model_A, model_B in tqdm(list(product(target_models, target_models))):
        if model_A not in data_dict or model_B not in model_dict:
            continue
        
        # 获取模型A的数据
        data = data_dict[model_A]
        # 获取模型B的判别器
        classifier = model_dict[model_B]
        
        # 进行预测
        with torch.no_grad():
            predictions = classifier(data['features'].to('cuda'))['logits']
            predictions = predictions.cpu().squeeze()
        
        # 计算准确率
        labels = data['labels']
        correct = ((predictions > 0.5).float() == labels).float().mean().item()
        
        results[(model_A, model_B)] = {
            'accuracy': correct,
            # 'predictions': predictions.numpy(),
            # 'labels': labels.numpy()
        }
    
    # 绘制热力图
    plot_accuracy_heatmap(results)
    
    return results

if __name__ == "__main__":
    results = cross_model_test()
    # 格式化打印结果
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