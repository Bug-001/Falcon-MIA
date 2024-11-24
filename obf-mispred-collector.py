import json
import os
import re
from pathlib import Path
from typing import Dict

def parse_folder_name(folder_name: str) -> Dict[str, str]:
    """Parse the folder name to extract task and dataset information."""
    # Extract only task and dataset parameters
    # XXX: More parameters can be added
    task_pattern = r"task\(([^)]+)\)"
    dataset_pattern = r"dataset\(([^)]+)\)"
    numdemo_pattern = r"num_demonstrations\(([^)]+)\)"
    
    task_match = re.search(task_pattern, folder_name)
    dataset_match = re.search(dataset_pattern, folder_name)
    numdemo_match = re.search(numdemo_pattern, folder_name)
    
    if task_match and dataset_match:
        return {
            "task_type": task_match.group(1),
            "dataset": dataset_match.group(1),
            "num_demo": int(numdemo_match.group(1)),
        }
    return None

def analyze_json_files(data_dir: Path):
    fp = []
    fn = []

    for root, dirs, files in os.walk(data_dir):
        # Skip if neither required file is present
        if not ('attack_results.json' in files and 'level_details.json' in files):
            continue
            
        # Get relative path to extract folder structure
        rel_path = os.path.relpath(root, data_dir)
        if rel_path == '.':
            continue
            
        # Parse folder name
        folder_info = parse_folder_name(os.path.basename(root))
        if not folder_info:
            continue

        root = Path(root)

        # Read metrics.json to get best_threshold
        with open(root/'metrics.json', 'r') as f:
            metrics = json.load(f)
            best_threshold = metrics[0]['best_threshold']

        # Read attack_results.json
        with open(root/'attack_results.json', 'r') as f:
            results = json.load(f)
            # Create a map of sample_id to prediction for faster lookup
            predictions = {item['sample_id']: item for item in results}

        # Read level_details.json and process
        with open(root/'level_details.json', 'r') as f:
            details = json.load(f)
        
        for detail in details:
            sample_id = detail['sample_id']
            similarity = detail['similarity']
            prediction = predictions[sample_id]
            
            # Check for misclassification conditions
            is_fp = similarity >= best_threshold and prediction['Membership'] is False
            is_fn = similarity < best_threshold and prediction['Membership'] is True
            
            info = {
                **folder_info,
                'sample_id': sample_id,
                'level': detail['level'],
                'obfuscated_text': detail['obfuscated_text'],
                'Original': prediction['Original'],
                'response': detail['response'],
                'mean_similarity': prediction['mean_similarity'],
                'similarity': similarity,
                'threshold': best_threshold,
            }

            if is_fp:
                fp.append(info)
            elif is_fn:
                fn.append(info)
    
    return fp, fn

def main():
    # Example usage
    
    input_dir = Path("cache/log/archived")
    output_dir = Path("cache/data/mispredicted")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/"false-positive").mkdir(parents=True, exist_ok=True)
    (output_dir/"false-negative").mkdir(parents=True, exist_ok=True)
    
    log_folders = [
        "Meta-Llama-3-8B-Instruct", # Uses 1 GPU by default
        "Qwen2.5-0.5B-Instruct",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-14B-Instruct",
        "Mistral-7B-Instruct-v0.2",
        # Add more models as needed
    ]
    
    for root_folder in log_folders:
        folder = input_dir/root_folder
        # Collect and process data
        fp, fn = analyze_json_files(folder)
        # Save the processed data
        with open(output_dir/"false-positive"/f"{root_folder}--output.json", 'w') as f:
            json.dump(fp, f, indent=2)
        print(f"Data saved to {output_dir/'false-positive'/root_folder}--output.json")
        with open(output_dir/"false-negative"/f"{root_folder}--output.json", 'w') as f:
            json.dump(fn, f, indent=2)
        print(f"Data saved to {output_dir/'false-negative'/root_folder}--output.json")

if __name__ == "__main__":
    main()