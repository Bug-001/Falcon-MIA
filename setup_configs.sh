#!/bin/bash

# setup_configs.sh
# This script copies configuration files from the configs directory to the root directory
# These configuration files can be modified by users to customize experiments

echo "Setting up configuration files..."

# Create .env file from template if it doesn't exist
if [ ! -f .env ]; then
  echo "Creating .env file from template..."
  cp configs/.env.example .env
  echo "Please edit .env file to add your API keys and other configurations"
fi

# Copy basic configuration files
echo "Copying basic configuration files..."
cp configs/data.yaml .
cp configs/query.yaml .
cp configs/attack_chat.yaml .
cp configs/server.yaml .
# # Copy experiment configuration files if they exist
# if [ -f configs/params.yaml ]; then
#   echo "Copying experiment configuration files..."
#   cp configs/params*.yaml .
#   cp configs/new-params*.yaml .
#   cp configs/params-mitigation*.yaml .
# fi

echo "Configuration setup complete!"
echo "You can now modify these configuration files in the root directory to customize your experiments."
echo "Remember to edit the .env file to add your API keys and other environment variables."

# Make the script executable
chmod +x setup_configs.sh 