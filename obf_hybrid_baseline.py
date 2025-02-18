import json
import os
import numpy as np
from typing import List, Dict, Tuple
import glob

def calculate_brainwash_accuracy_with_threshold(results: List[Dict], threshold: float) -> float:
    """基于阈值计算brainwash attack的准确率"""
    correct = 0
    total = len(results)
    
    for item in results:
        # 使用Average Iterations作为判断依据
        predicted_member = item["Average Iterations"] >= threshold
        if predicted_member == item["Is member"]:
            correct += 1
            
    return correct / total

def find_best_threshold(results: List[Dict]) -> Tuple[float, float]:
    """找到最优阈值及其对应的准确率"""
    # 获取所有可能的Average Iterations值
    thresholds = sorted(set(item["Average Iterations"] for item in results))
    
    best_acc = 0
    best_threshold = thresholds[0]
    
    # 遍历所有可能的阈值
    for threshold in thresholds:
        acc = calculate_brainwash_accuracy_with_threshold(results, threshold)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
            
    return best_threshold, best_acc

def process_model_results(model_name: str, seeds: List[int]) -> Dict[str, List[float]]:
    """处理单个模型的所有实验结果"""
    brainwash_accs = []
    hybrid_accs = []
    
    for seed in seeds:
        # 使用glob模糊匹配文件夹路径
        pattern = f"cache/log/*{model_name}*/task(default)--dataset(trec)--num_demonstrations(3)--type(Hybrid)--random_seed({seed})*"
        matching_dirs = glob.glob(pattern)
        
        if matching_dirs:
            base_path = matching_dirs[0]  # 取第一个匹配的路径
            
            # 读取brainwash结果
            brainwash_path = os.path.join(base_path, "brainwash-attack_results.json")
            if os.path.exists(brainwash_path):
                with open(brainwash_path, 'r') as f:
                    results = json.load(f)
                    _, best_acc = find_best_threshold(results)
                    brainwash_accs.append(best_acc)
            
            # 读取hybrid结果
            hybrid_path = os.path.join(base_path, "metrics.json")
            if os.path.exists(hybrid_path):
                with open(hybrid_path, 'r') as f:
                    metrics = json.load(f)
                    acc = metrics[0]["accuracy"]
                    hybrid_accs.append(acc)
    
    return {
        "brainwash": brainwash_accs,
        "hybrid": hybrid_accs
    }

def main():
    models = ['gpt-4o-mini', 'gpt-3.5-turbo', 'Ministral-8B', 'Qwen-7B', 'Llama-3-8B', 'Llama-2-7B']
    seeds = [124, 125, 126]
    
    print("模型性能评估结果:")
    print("-" * 80)
    print(f"{'Model':<15} {'Brainwash (mean±std)':<25} {'Hybrid (mean±std)':<25}")
    print("-" * 80)
    
    for model in models:
        results = process_model_results(model, seeds)
        
        # 计算统计量
        brainwash_mean = np.mean(results["brainwash"]) if results["brainwash"] else 0
        brainwash_std = np.std(results["brainwash"]) if results["brainwash"] else 0
        hybrid_mean = np.mean(results["hybrid"]) if results["hybrid"] else 0
        hybrid_std = np.std(results["hybrid"]) if results["hybrid"] else 0
        
        # 格式化输出
        print(f"{model:<15} {brainwash_mean:.3f}±{brainwash_std:.3f} {' '*10} {hybrid_mean:.3f}±{hybrid_std:.3f}")

if __name__ == "__main__":
    main()
