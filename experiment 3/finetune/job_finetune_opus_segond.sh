#!/bin/bash
#SBATCH --account=def-richy
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/chaouin/projects/def-richy/chaouin/coptic-fr-translation/logs/fine_tune_opus_segond_%j.out
#SBATCH --error=/home/chaouin/projects/def-richy/chaouin/coptic-fr-translation/logs/fine_tune_opus_segond_%j.err
#SBATCH --mail-user=nasma.chaoui.1@ulaval.ca
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Début du job à $(date)"
echo "Job lancé sur $(hostname)"

source ~/py311/bin/activate
nvidia-smi

# Aller dans le répertoire du script
cd ~
cd "projects/def-richy/chaouin/coptic-fr-translation/experiment 3/finetune"
echo "Répertoire courant : $(pwd)"

# Lancer l'entraînement
python "$(pwd)/finetune_opus_coptic_fr.py" \
  --data_path "/home/chaouin/projects/def-richy/chaouin/data/train_segond_data.csv" \
  --output_dir "opus-finetuned-coptic-fr-segond-data" \

echo "Fin du job à $(date)"
