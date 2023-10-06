#!/bin/bash

# git clone git@github.com:alirezabayatmk/lewis_final.git
# chmod +x custom_scripts/installation_process.sh
# run this script from the root of the project

echo "Starting LEWIS environment setup..."

#############################################################################################################
#                                           creating the miniconda env                                      #
#############################################################################################################
#replace when using LUIS cluster
echo "Creating conda environment..."
conda create -y -n lewis python=3.8

echo "Activating conda environment..."
conda activate lewis


#############################################################################################################
#                                         installing required packages                                      #
#############################################################################################################
echo "Installing ipykernel..."
conda install -y ipykernel

echo "Installing PyTorch and related packages..."
conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

echo "Installing additional Python packages..."
pip install pandas transformers python-Levenshtein


#############################################################################################################
#                                    installing NVIDIA APEX for speedup                                     #
#############################################################################################################
echo "Installing NVIDIA apex..."
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
cd ..
echo "NVIDIA apex installation completed."


#############################################################################################################
#                                  setting up fairseq (LEWIS Author's fork)                                 #
#############################################################################################################
echo "Cloning fairseq (Author's fork)..."
git clone https://github.com/machelreid/fairseq
cd fairseq/

# Installing fairseq
echo "Installing fairseq..."
pip install --editable ./
pip install sacrebleu sacremoses tensorboardX
cd ..
echo "Fairseq installation completed."

# copying fairseq to lewis_final
cp -r fairseq/ lewis_final/fairseq
echo "Lews cloning and setup completed."


#############################################################################################################
#                                 preparing dataset to match LEWIS format                                   #
#############################################################################################################
mkdir downloaded_data
echo "Created directory 'downloaded_data'"

cd downloaded_data || { echo "Failed to change directory."; exit 1; }
echo "Changed directory to 'downloaded_data'"

mkdir cmv
echo "Created directory 'cmv'"

cd cmv || { echo "Failed to change directory."; exit 1; }
echo "Changed directory to 'cmv'"

mkdir dm1-app
echo "Created directory 'dm1-app'"

mkdir dm2-inapp
echo "Created directory 'dm2-inapp'"

python custom_scripts/data_preparation.py
echo "Ran data_preparation.py"


#############################################################################################################
#                                       downloading pre-trained models                                      #
#############################################################################################################
cd downloaded_data || { echo "Failed to change directory."; exit 1; }
echo "Changed directory to 'downloaded_data'"

echo "Downloading pre-trained BART..."
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz
tar -xvzf bart.base.tar.gz

echo "Downloading pre-trained RoBERTa..."
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
tar -xvzf roberta.base.tar.gz

echo "Downloading tokenizer for RoBERTA (which uses GPT-2 BPE)..."
mkdir gpt2_bpe
wget -P gpt2_bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json gpt2_bpe
wget -P gpt2_bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe gpt2_bpe/
wget -P gpt2_bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt


#############################################################################################################
#                                      running LEWIS scripts in a row                                       #
#############################################################################################################
echo "Running preprocess-roberta-classifier.sh..."
source roberta-classifier/preprocess-roberta-classifier.sh downloaded_data cmv dm1-app dm2-inapp

echo "Running train-roberta-classifier.sh..."
source roberta-classifier/train-roberta-classifier.sh downloaded_data cmv

echo "Running convert_roberta_original_pytorch_checkpoint_to_pytorch.py..."
python roberta-classifier/convert_roberta_original_pytorch_checkpoint_to_pytorch.py --roberta_checkpoint_path downloaded_data/roberta-classifier/cmv/checkpoints --pytorch_dump_folder_path downloaded_data/roberta-classifier/cmv --classification_head classification_head

echo "Running preprocess-bart-denoising.sh..."
source bart-denoising/preprocess-bart-denoising.sh downloaded_data cmv dm1-app
source bart-denoising/preprocess-bart-denoising.sh downloaded_data cmv dm2-inapp

echo "Running train-bart-denoising.sh..."
source bart-denoising/train-bart-denoising.sh downloaded_data cmv dm1-app
source bart-denoising/train-bart-denoising.sh downloaded_data cmv dm2-inap

echo "Running get_synthesized_data.py..."
python get_synthesized_data.py --d1_model_path downloaded_data/bart-denoising/cmv/dm1-app/checkpoints/checkpoint_best.pt --d2_model_path downloaded_data/bart-denoising/cmv/dm2-inapp/checkpoints/checkpoint_best.pt --d1_file downloaded_data/cmv/dm1-app/train.txt --d2_file downloaded_data/cmv/dm2-inapp/train.txt --out_file file.json --hf_dump downloaded_data/roberta-classifier/cmv

