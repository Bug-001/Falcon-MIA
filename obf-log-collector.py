import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import re

class LogCollector:
    def __init__(self, output_dir: str = "~/.cache/better-mia/data"):
        self.output_dir = os.path.expanduser(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def parse_folder_name(self, folder_name: str) -> Dict[str, str]:
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

    def process_attack_results(self, file_path: str) -> Dict[int, Dict]:
        """Process attack_results.json and return a dictionary keyed by sample_id."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return {
            item["sample_id"]: {
                "original": item["Original"],
                "membership": item["Membership"]
            }
            for item in data
        }

    def process_level_details(self, file_path: str) -> Dict[int, str]:
        """Process level_details.json and return a dictionary keyed by sample_id."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return {
            (item["sample_id"], item["level"]): {
                "response": item["response"],
            }
            for item in data
        }

    def collect_data(self, root_folder: str) -> List[Dict]:
        """Main function to collect and process all data."""
        collected_data = []
        
        # Walk through all directories
        for root, dirs, files in os.walk(root_folder):
            # Skip if neither required file is present
            if not ('attack_results.json' in files and 'level_details.json' in files):
                continue
                
            # Get relative path to extract folder structure
            rel_path = os.path.relpath(root, root_folder)
            if rel_path == '.':
                continue
                
            # Parse folder name
            folder_info = self.parse_folder_name(os.path.basename(root))
            if not folder_info:
                continue
            
            # Process both JSON files
            attack_results = self.process_attack_results(os.path.join(root, 'attack_results.json'))
            level_details = self.process_level_details(os.path.join(root, 'level_details.json'))
            
            # Merge data
            for (sample_id, level), level_data in level_details.items():
                if sample_id in attack_results:
                    entry = {
                        "model": os.path.basename(root_folder),  # Top-level folder name is model name
                        "task_type": folder_info["task_type"],
                        "dataset": folder_info["dataset"],
                        "num_demo": folder_info["num_demo"],
                        "sample_id": sample_id,
                        "original_text": attack_results[sample_id]["original"],
                        "membership": attack_results[sample_id]["membership"],
                        "response": level_data["response"],
                        "level": level
                    }
                    collected_data.append(entry)
        
        return collected_data

    def save_data(self, data: List[Dict], filename: str = "collected_data.json"):
        """Save collected data to the output directory."""
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Data saved to {output_path}")

def main():
    # Example usage
    collector = LogCollector()
    
    root_folders = [
        "Meta-Llama-3-8B-Instruct", # Uses 1 GPU by default
        "Qwen2.5-0.5B-Instruct",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-14B-Instruct",
        "Mistral-7B-Instruct-v0.2",
        # Add more models as needed
    ]
    
    for root_folder in root_folders:
        folder = os.path.join("output", root_folder)
        # Collect and process data
        collected_data = collector.collect_data(folder)
        # Save the processed data
        collector.save_data(collected_data, f"{root_folder}--output.json")

if __name__ == "__main__":
    main()