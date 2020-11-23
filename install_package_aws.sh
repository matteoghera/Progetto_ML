#!/bin/bash

set -e

# OVERVIEW
# This script installs a single pip package in a single SageMaker conda environments.

sudo -u ec2-user -i <<'EOF'
# PARAMETER
ENVIRONMENT=pytorch_p36

source /home/ec2-user/anaconda3/bin/activate "$ENVIRONMENT"
conda install pip --name "$ENVIRONMENT" --yes

#pip install -r /home/ec2-user/SageMaker/Progetto_ML/requirements.txt
pip install psutil
pip install path
pip install seaborn
pip install scikit-learn
pip install pandas
pip install matplotlib
pip install scipy

git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .

source /home/ec2-user/anaconda3/bin/deactivate
EOF