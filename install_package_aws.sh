#!/bin/bash

set -e

# OVERVIEW
# This script installs a single pip package in a single SageMaker conda environments.
# chmod +x install_package_aws.sh

sudo -u ec2-user -i <<'EOF'
# PARAMETER
ENVIRONMENT=pytorch_p36

source /home/ec2-user/anaconda3/bin/activate "$ENVIRONMENT"

git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .

source /home/ec2-user/anaconda3/bin/deactivate
EOF