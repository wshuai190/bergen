#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --job-name=TREC_RAG
#SBATCH --partition=gpu_cuda
#SBATCH --account=a_ielab
#SBATCH --gres=gpu:h100:1
#SBATCH --time=10:00:00
#SBATCH -o /scratch/project/neural_ir/dylan/COCOM/print.txt
#SBATCH -e /scratch/project/neural_ir/dylan/COCOM/error.txt




module load anaconda3/2022.05
module load cuda
source activate bergen

cd ..

HYDRA_FULL_ERROR=1 python3 bergen.py dataset="msmarco_dev_2_1" retriever="spladev3" generator="tinyllama-chat"


