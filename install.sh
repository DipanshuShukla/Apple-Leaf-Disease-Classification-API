#!/bin/bash

# Download the Anaconda installer
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh

# Install Anaconda, saying yes to all prompts
bash Anaconda3-2022.10-Linux-x86_64.sh -b -f -p $HOME/anaconda3

# Add Anaconda to the system PATH
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> $HOME/.bashrc
source $HOME/.bashrc

# Install required packages
conda install -y -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# Restart the terminal
exec bash

# Install Python packages from requirements.txt
pip3 install -r requirements.txt

# Run the API runner
python3 api_runner.py
