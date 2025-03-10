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

## Usage

### Basic Usage

The main module is `icl.py`, which can be run with different configuration files:

```bash
python icl.py --data data.yaml --attack attack_chat.yaml --query query.yaml
```

### Configuration Files

- `data.yaml`: Specifies the dataset and task settings
- `attack_chat.yaml`: Defines the attack strategy and parameters
- `query.yaml`: Configures the target model and query settings

### Running Experiments

To run multiple experiments with different configurations, use `run.py`:

```bash
python run.py --config params.yaml
```

## Reproducing Paper Experiments

We designed a experiment framework to facilitate the research. Following are the steps to reproduce the paper experiments, while you can learn how to use this framework to design your own experiments.

(Under construction below)

### Main Results (Table 2)

To reproduce the main results comparing FALCON with baseline methods across different models and datasets:

```bash
# Run FALCON with Obfuscation attack
python run.py --config params-1.yaml

# Run baseline methods (GAP, Inquiry, Repeat, Brainwash, Hybrid)
python run.py --config params-2.yaml
```

### Influence of Demonstrations (Figure 4)

To evaluate how the number of demonstrations affects attack performance:

```bash
# Vary the number of demonstrations from 1 to 10
python run.py --config new-params-1.yaml
```

### Influence of Model Size (Figure 5)

To analyze the correlation between model scale and attack efficacy:

```bash
# Test across different model sizes (0.5B to 72B)
python run.py --config new-params-2.yaml
```

### Ablation Studies (Table 3 & 4)

To understand the contribution of different obfuscation techniques:

```bash
# Ablation study on obfuscation techniques
python obf-ablation.py --config new-params-3.yaml

# Ablation study on IDF-weighted similarity measures
python obf-ablation.py --config new-params-4.yaml
```

### Cross-Model Testing (Figure 6)

To evaluate the transferability of the attack across different models:

```bash
python obf_cross_model_test.py --config params-mitigation-llama3.yaml
```

### Mitigation Strategies (Table 5)

To evaluate potential defense mechanisms:

```bash
python run.py --config params-mitigation-qwen.yaml
python run.py --config params-mitigation-llama3.yaml
```

## Result Analysis

After running experiments, you can analyze the results using the provided scripts:

```bash
# Collect and analyze metrics
python obf_metrics_getter.py --results_dir results/

# Comprehensive result analysis
python obf-result-analysis.py --results_dir results/
```

## Framework Structure

- `attack/`: Contains implementations of different attack strategies
  - `__init__.py`: Base attack strategy class and factory method
  - `obfuscation.py`: FALCON's obfuscation attack implementation
  - `gap.py`, `inquiry.py`, `repeat.py`, `brainwash.py`, `hybrid.py`: Baseline methods
  - `mitigation.py`: Defense mechanisms
- `string_utils.py`: Text obfuscation techniques
- `data.py`: Dataset loading and processing
- `utils.py`: Utility functions
- `detector.py`: Neural membership detector implementation
- `configs/`: Contains default configuration files
  - `data.yaml`: Default dataset configuration
  - `query.yaml`: Default model query configuration
  - `attack_chat.yaml`: Default attack configuration
  - `.env.example`: Template for environment variables

## Citation

If you use this code in your research, please cite our paper:

```
@article{su2024falcon,
  title={FALCON: A Universal Text-only Membership Inference Attack Framework against In-context Learning},
  author={Su, Haitao and Qin, Yue and Zhou, Yuan},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2024}
}
```

## License

[MIT License](LICENSE) 