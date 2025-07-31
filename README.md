# 🏺 Coptic–French Neural Machine Translation (NMT)

This repository contains the full codebase used for the experiments presented in the paper:

**Neural Machine Translation for Coptic–French: Strategies for Low-Resource Ancient Languages**  
**📄 [Read the full article - TBA]()**

## 📦 Project Overview

This project investigates various neural machine translation strategies for translating Coptic into French, focusing on four major questions:

1. **Direct vs Pivot Translation**  
2. **Pre-trained Model Choice**
3. **Multi-version Fine-Tuning**
4. **Robustness to Manuscript Noise**

Two final models were trained and released:
- A multilingual translation model based on OPUS Helsinki
- A transfer learning model based on hieroglyphic pretraining

---

## 📁 Dataset Access

All required evaluation and training datasets (CSV/JSON) are hosted on [Hugging Face Datasets Hub](https://huggingface.co/datasets/chaouin/coptic-french-translation-data).

To fetch them locally, run:

```bash
python download_data.py
```

This script downloads **all files**, including:
- Training data (`experiment */data/`)
- Generated translations
- Evaluation metrics
- Multi-version aligned corpora

---

## 🧪 Main Scripts

| Script | Purpose                                                                                                       |
|--------|---------------------------------------------------------------------------------------------------------------|
| [`download_data.py`](./download_data.py) | Fetches **all data files** from Hugging Face used in this project.                                            |
| [`generate_translation_helsinki.py`](./generate_translation_helsinki.py) | Generates translations using the **Coptic–French model** fine-tuned from the **Helsinki multilingual model**. |
| [`generate_translation_hiero.py`](./generate_translation_hiero.py) | Generates translations using the **Coptic–French model** fine-tuned from the **Hieroglyphic-based model**.    |

## 🧠 Published Models

The following Hugging Face models were published based on our findings:
- [coptic-french-translation-helsinki](https://huggingface.co/chaouin/coptic-french-translation-helsinki)
- [coptic-french-translation-hiero](https://huggingface.co/chaouin/coptic-french-translation-hiero)

---

## 🔬 Models Used in Experiments

The following models were used and/or fine-tuned during our four experiments:
- [Helsinki-NLP/opus-mt-tc-bible-big-mul-mul](https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-mul-mul)
- [mattiadc/hiero-transformer](https://huggingface.co/mattiadc/hiero-transformer)
- [megalaa/coptic-english-translator](https://huggingface.co/megalaa/coptic-english-translator)
- [google-t5/t5-base](https://huggingface.co/google-t5/t5-base)

---

## 📊 Evaluation Metrics

Evaluations are performed using:
- **BERTScore**
- **BLEURT**
- **COMET**
- **METEOR**

These metrics are computed for all translation strategies and model variations across clean and noisy data.

---

## 📚 Citation

If you use this code or dataset, please cite our paper:

```
TBA
```
