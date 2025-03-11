# FALCON: A Universal Text-only Membership Inference Attack Framework

This repository contains the implementation of FALCON (Flexible Attack on Language Context via Obfuscation), a novel membership inference attack framework for in-context learning systems. FALCON reformulates membership inference as a text de-obfuscation task, inspired by cognitive psychology principles.

## Overview

FALCON is the first task-agnostic membership inference attack framework for in-context learning systems. It leverages a previously unexplored signal: language models exhibit stronger de-obfuscation capabilities for text sequences they have previously encountered in their context window.

The framework consists of three main components:
1. **Text Obfuscation**: Multiple techniques to transform original text
2. **Stealthy Querying**: Carefully designed prompts to elicit differential responses
3. **Neural Membership Detector**: A classifier that synthesizes various similarity signals

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Bug-001/Better-MIA.git
cd Better-MIA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up configuration files:
```bash
# Run the setup script to copy configuration files from configs directory
bash setup_configs.sh
```
This script copies necessary configuration files from the `configs` directory to the root directory. These files can be modified to customize your experiments.

4. Set up environment variables. Edit the `.env` file created by the setup script to include your API keys and other required configurations.

## Basic Usage

The main module is `icl.py`, which can be run with different configuration files:

```bash
python icl.py --data data.yaml --attack attack_chat.yaml --query query.yaml
```

### Data Configuration

The `data.yaml` file contains demonstrations settings of the target model:

```yaml
dataset: "gpqa"
task: "default"
num_demonstrations: 3
```

### Query Configuration

The `query.yaml` file contains important settings for the model you want to use:

```yaml
# For remote models (like OpenAI, Anthropic, etc.)
# model_type: openai
# model: gpt-3.5-turbo
# base_url: https://api.openai.com/v1

# For local models
model_type: local
model: server.yaml  # Path to server configuration file

# See detailed text generation settings in the `query.yaml` file
```

When using a local model (`model_type: local`), the `model` parameter should point to a server configuration file (e.g., `server.yaml`) that contains the model settings:

```yaml
# Locally deployed server configuration example
model-tag: "mistralai/Ministral-8B-Instruct-2410"  # HuggingFace model identifier
host: "192.168.241.6"  # Server IP address
port: 8000  # Server port
trust-remote-code: false
dtype: "auto"
gpu-memory-utilization: 0.90
tensor-parallel-size: 1

# All the parameters used by vllm.server module can be applied here
```

While the attack framework only use the `model-tag`, `host`, and `port` parameters in `server.yaml` for local model querying, this server configuration can be used to deploy a local model server with vLLM by the following command:

```bash
python llm/server.py -c server.yaml
```

### Attack Configuration

The `attack_chat.yaml` file defines the attack strategy and its parameters:

```yaml
# Example attack_chat.yaml
attack_type: "Obfuscation"  # Attack type (obfuscation, gap, inquiry, etc.)
obfuscation_methods: "character_swap"  # For obfuscation attack
random_seed: 125

# Much more details can be found in the `attack_chat.yaml` file
```

You can customize the attack by modifying these parameters or implementing your own attack strategy in the `attack/` directory.

## Batch Experiment Framework

To run multiple experiments with different configurations, use `run.py`:

```bash
python run.py --config params.yaml
```

`params.yaml` is a batch experiment configuration file that allows you to run multiple experiment groups in a single execution. The structure of this file is as follows:

```yaml
# Experiment name prefix
name_prefix: "Qwen-7B-demo-location"

# Program/module to run
program:
  module: "icl"          # Program module
  function: "main"       # Main function name

# Parallel strategy configuration
parallel:
  enable: true           # Whether to enable parallel execution
  max_workers: 16        # Maximum number of parallel processes

# Parameter configuration, which order is consistent with `icl.py`
params:
  - param: "data.yaml"   # Data configuration file
    config:
      dataset:           # Dataset parameters
        - value: lexglue
          dependency:
            task: [judgment, generation]
      num_demonstrations: [5]  # Number of examples
  
  - param: "attack_chat.yaml"  # Attack configuration file
    config:
      technique: ["character_swap", "synonym_replacement"]  # Different attack techniques
      selected_attack_sample: [1, 2, 3, 4, 5]  # Selected attack samples
  
  - param: "query.yaml"  # Query configuration file
    config:
      model: ["server.yaml"]  # Model configuration
```

With this approach, you can easily configure multiple combinations of experiment parameters, and the system will automatically create separate experiments for each combination. For example, the configuration above will create separate experiments for each attack technique and each selected attack sample, greatly simplifying the setup and execution process of batch experiments.

## Result Post-processing Scripts

After running experiments, FALCON provides several scripts to process and analyze the results:

### 1. Combining Multiple Obfuscation Techniques

The `obf_multi_techniques.py` script combines results from different obfuscation techniques to create a more robust attack. This script analyzes the similarities generated by different techniques and merges them into a single dataset.

```bash
# Combine results from different obfuscation techniques
python obf_multi_techniques.py --dir "Meta-Llama-3-8B-Instruct"
```

This script:
- Collects similarity data from experiments with different obfuscation techniques
- Merges the similarity features from each technique (prefixing them with technique identifiers)
- Creates a new experiment with the combined features
- Runs the membership detector on the combined dataset

The combined results typically show higher accuracy than individual techniques, demonstrating the complementary nature of different obfuscation methods.

### 2. Collecting Metrics Across Experiments

The `obf_metrics_getter.py` script collects and summarizes metrics from multiple experiments, making it easy to compare performance across different configurations. Example:

```bash
# Collect metrics from experiments with a specific prefix
python obf_metrics_getter.py --dir "Meta-Llama-3-8B-Instruct"
python obf_metrics_getter.py --dir "Meta-Llama-3-8B-Instruct/obf_technique_test"
```

This script:
- Traverses the results directory to find all experiment folders
- Extracts accuracy metrics from each experiment
- Calculates mean and standard deviation for experiments with multiple seeds
- Sorts and displays results in descending order of accuracy

The output format is `experiment_name: accuracy±std_deviation`, making it easy to identify the best-performing configurations.

## Example Workflow

A typical post-processing workflow might look like:

1. Prepare the basic configuration files, i.e. `data.yaml`, `attack_chat.yaml`, `query.yaml` and `server.yaml` if you want to use local model.

2. Run multiple experiments with different configurations:
```bash
python run.py --config params.yaml
```

3. Combine results from different obfuscation techniques:
```bash
python obf_multi_techniques.py --dir "Meta-Llama-3-8B-Instruct"
```

4. Collect and analyze metrics across all experiments:
```bash
python obf_metrics_getter.py --dir "Meta-Llama-3-8B-Instruct/obf_technique_test"
```

The results can then be visualized using standard data analysis tools like Matplotlib.

## Customizing Experiments

To design your own experiments:

1. **Create custom configuration files**:
   - Modify existing YAML files in the `configs/` directory
   - Create new configuration files for specific experiment settings

2. **Implement custom attack strategies**:
   - Extend the base attack class in `attack/__init__.py`
   - Implement your attack logic in a new file in the `attack/` directory
   - Register your attack in the attack factory method

3. **Add custom datasets**:
   - Extend the data loader in `data.py`
   - Implement preprocessing for your dataset

<!-- ## Citation

If you use this code in your research, please cite our paper:

```
@article{su2024falcon,
  title={FALCON: A Universal Text-only Membership Inference Attack Framework against In-context Learning},
  author={Su, Haitao and Qin, Yue and Zhou, Yuan},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2024}
}
``` -->

## License

[MIT License](LICENSE)