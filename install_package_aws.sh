#!/bin/bash

set -e

# OVERVIEW
# This script installs a single pip package in a single SageMaker conda environments.

sudo -u ec2-user -i <<'EOF'
# PARAMETER
ENVIRONMENT=pytorch_p36

source /home/ec2-user/anaconda3/bin/activate "$ENVIRONMENT"
conda install pip --name "$ENVIRONMENT" --yes

pip install -r requirements.txt

source /home/ec2-user/anaconda3/bin/deactivate
EOF