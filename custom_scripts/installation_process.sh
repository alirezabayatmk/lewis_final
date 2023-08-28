#!/bin/bash

echo "Starting environment setup..."

# Creating environment
echo "Creating conda environment..."
conda create -y -n lewis python=3.8

echo "Activating conda environment..."
conda activate lewis

# Installing packages
echo "Installing ipykernel..."
conda install -y ipykernel

echo "Installing PyTorch and related packages..."
conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

echo "Installing additional Python packages..."
pip install pandas transformers python-Levenshtein

# Installing NVIDIA apex
echo "Installing NVIDIA apex..."
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
cd ..
echo "NVIDIA apex installation completed."

# Cloning fairseq (Author's fork)
echo "Cloning fairseq (Author's fork)..."
git clone https://github.com/machelreid/fairseq
cd fairseq/

# Installing fairseq
echo "Installing fairseq..."
pip install --editable ./
pip install sacrebleu sacremoses tensorboardX
cd ..
echo "Fairseq installation completed."

# Cloning lews (Alireza's version)
echo "Cloning lews (Alireza's version)..."
git clone git@github.com:alirezabayatmk/lewis.git
cp -r fairseq/ lewis/fairseq
echo "Lews cloning and setup completed."

echo "Environment setup completed."
