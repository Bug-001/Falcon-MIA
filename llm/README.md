# General-Purpose LLM Tool

This repository contains a versatile tool for working with Large Language Models (LLMs). It includes functionality for fine-tuning, querying, and serving LLMs.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Fine-tuning](#fine-tuning)
   - [Querying](#querying)
   - [Serving](#serving)

## Overview

This repository contains a versatile and comprehensive tool for working with Large Language Models (LLMs). It provides a unified interface for three core functionalities:

1. **Fine-tuning**: Customize pre-trained models on specific datasets to enhance performance for targeted tasks.
   - Supports various Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA, Prefix Tuning, P-Tuning, and IA3.
   - Compatible with both local transformer models and remote API-based models.
   - Configurable training parameters and data preprocessing options.

2. **Querying**: Interact with LLMs through a flexible query interface.
   - Supports multiple model types including OpenAI, local models, and Infinigence.
   - Enables chat-based interactions with predefined and dynamic responses.
   - Allows for custom instructions and fine-grained control over generation parameters.
   - Offers both interactive and non-interactive querying modes.
   - Supports full and partial output mode.

3. **Serving**: Deploy fine-tuned or pre-trained models for efficient inference.
   - Utilizes VLLM for high-performance model serving.
   - Provides customizable server configurations for optimal resource utilization.
   - Automatically selects appropriate chat templates based on the model architecture.
   - Supports various model types with extensible template system.

Key Features:
- Unified YAML-based configuration for all functionalities.
- Modular design allowing for easy extension and customization.
- Comprehensive logging and error handling for robust operation.
- Support for multiple GPU configurations and quantization options.
- Integration with popular LLM platforms and local deployment options.

This tool is designed to streamline the entire LLM workflow, from model customization to deployment and interaction, catering to researchers, developers, and organizations working with state-of-the-art language models.
## Installation

[Instructions for installing the tool and its dependencies]

## Usage

### Fine-tuning

To fine-tune a model, use the `fine_tune.py` script with a configuration file:

```
python fine_tune.py --config path/to/your/config.yaml
```

The fine-tuning process includes:
- Loading and preprocessing the dataset
- Initializing the model and tokenizer
- Applying PEFT (Parameter-Efficient Fine-Tuning) methods
- Training the model using the SFTTrainer
- Saving the fine-tuned model and tokenizer

The `fine-tune.yaml` file contains all the necessary configurations for fine-tuning:

- `id`: Unique identifier for the experiment
- `model`: Specifications for the model, including type, name, and quantization options
- `data`: Dataset configuration, including name, prompt template, and preprocessing options
- `training`: Training parameters such as output directory, batch size, learning rate, etc.
- `peft`: Configuration for Parameter-Efficient Fine-Tuning methods
- `hardware`: Hardware-related settings
- `remote_model`: Configuration for remote API-based models (if applicable)

### Querying

To query a model, use the `query.py` script with a configuration file:

```
python query.py --config path/to/your/config.yaml [--full]
```

The `query.yaml` file contains all the necessary configurations for querying:

- `model_type`: Type of model to use (openai, local, infinigence)
- `model`: Specific model to use
- `query_type`: Type of query (chat, instruction)
- `temperature`, `max_tokens`, `top_p`, etc.: Model generation parameters
- `instruction`: Custom instruction for the model
- `chats`: Predefined chat sequences with system, user, and assistant messages

### Serving

To serve a model, use the `server.py` script with a configuration file:

```
python server.py --config path/to/your/server.yaml
```

The serving process includes:
- Loading a specified model
- Setting up a VLLM server with custom configurations
- Automatically selecting an appropriate chat template based on the model name

The `server.yaml` file contains the necessary configurations for serving a model:

- `model-tag`: The identifier for the model to be served
- `host`: The host address for the server (default: "127.0.0.1")
- `port`: The port number for the server
- `trust-remote-code`: Whether to trust remote code (boolean)
- `dtype`: Data type for model computations (e.g., "auto")
- `gpu-memory-utilization`: GPU memory utilization factor (0.0 to 1.0)
- `max-num-seqs`: Maximum number of sequences to process
- `quantization`: Quantization settings (if applicable)

The server automatically selects an appropriate chat template based on the model name. Templates are stored in the `templates/` directory and include:

- `llama.jinja`: For LLaMA models
- `claude.jinja`: For Claude models
- `mistral.jinja`: For Mistral models
- `general.jinja`: Default template for other models

You can add custom templates to support additional model types.

**Usage Notes**

- The server uses VLLM to efficiently serve large language models.
- Custom VLLM arguments can be passed directly to the `server.py` script.
- The server automatically selects the appropriate chat template based on the model name, ensuring compatibility with different model architectures.
