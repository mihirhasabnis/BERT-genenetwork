#!/bin/bash
#SBATCH --job-name=bertwalk
#SBATCH --output=bertwalk.out
#SBATCH --error=bertwalk.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=40
#SBATCH --nodelist=cn12

module load anaconda3-2023.3
python Bertwalk_gene.py