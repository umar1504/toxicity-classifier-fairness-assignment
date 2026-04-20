# Assignment 2: Toxicity Classifier with Fairness & Robustness

## Student Information
- **Course**: Responsible & Explainable AI
- **Assignment**: 2 - Toxicity Classification
- **Student Name**: [Your Name]
- **Python Version**: 3.12
- **GPU Used**: Tesla T4 (Google Colab)

## Project Overview
This project implements a toxicity classifier using DistilBERT, with:
- Bias audit across demographic groups
- Adversarial attack testing (evasion + poisoning)
- Fairness mitigation techniques
- Production-ready guardrail pipeline

## Results Summary

| Part | Key Result |
|------|------------|
| Part 1 | Accuracy: 94.56%, AUC: 0.9478 |
| Part 2 | Disparate Impact: 1.146 |
| Part 3 | Evasion ASR: 99.4% |
| Part 4 | Fairness constraints incompatible |
| Part 5 | Auto-action F1: 0.6165 |

## Key Findings
1. **Evasion attacks** bypass the model 99.4% of the time
2. **Systematic bias** against Black-associated comments (1.14x higher FPR)
3. **Fairness constraints** cannot be simultaneously satisfied when base rates differ
4. **3-layer pipeline** effectively routes uncertain cases to human review

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
