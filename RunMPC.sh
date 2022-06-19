#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --time=14-00:00:00
#SBATCH --output=MPC-Neptune-Test-Log.out
#SBATCH --job-name='Multi Probalistic Composer'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tun84049@temple.edu
cd /home/tun84049/MCP-Hotels8k-Compatible/code
python main.py --config_name=probabilistic.yaml --mode=train