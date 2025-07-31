#!/bin/bash
#SBATCH --account=def-richy
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-05:00:00
#SBATCH --output=/home/chaouin/projects/def-richy/chaouin/coptic-fr-translation/logs/fine_tune_opus_noisy_30_%j.out
#SBATCH --error=/home/chaouin/projects/def-richy/chaouin/coptic-fr-translation/logs/fine_tune_opus_noisy_30_%j.err
#SBATCH --mail-user=nasma.chaoui.1@ulaval.ca
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Début du job à $(date)"
echo "Job lancé sur $(hostname)"

source ~/py311/bin/activate
nvidia-smi

# Aller dans le répertoire du script
cd ~
cd "projects/def-richy/chaouin/coptic-fr-translation/experiment 4/finetune"
echo "Répertoire courant : $(pwd)"

# Lancer l'entraînement
python "$(pwd)/finetune_opus_coptic_fr.py" \
  --data_path "/home/chaouin/projects/def-richy/chaouin/data/train_noisy_30_data.csv" \
  --output_dir "opus-finetuned-coptic-fr-noisy-30-data" \

echo "Fin du job à $(date)"
