#!/bin/bash
#SBATCH --account=def-richy
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-08:00:00
#SBATCH --output=/home/chaouin/projects/def-richy/chaouin/coptic-fr-translation/logs/exp4_evaluate_translations_bleurt_opus_%j.out
#SBATCH --error=/home/chaouin/projects/def-richy/chaouin/coptic-fr-translation/logs/exp4_evaluate_translations_bleurt_opus_%j.err
#SBATCH --mail-user=nasma.chaoui.1@ulaval.ca
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Début du job à $(date)"
echo "Job lancé sur $(hostname)"

source ~/py311/bin/activate
nvidia-smi

# Aller dans le répertoire du script
cd ~
cd "projects/def-richy/chaouin/coptic-fr-translation/experiment 4/evaluation"
echo "Répertoire courant : $(pwd)"

# Lancer l'entraînement
python "$(pwd)/evaluate_coptic_fr_translations_comparison_bleurt.py"

echo "Fin du job à $(date)"
