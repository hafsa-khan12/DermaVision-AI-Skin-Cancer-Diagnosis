
# Phase 5 - Final Evaluation, Robustness and Error Analysis

This folder contains Phase 5 work for DermaVision AI.

## Phase 5 Focus

- Final model evaluation
- Classification report
- Confusion matrix analysis
- Robustness testing
- Error analysis
- Latency and efficiency profiling
- Statistical summary
- Baseline vs final model comparison

## Final Model

The selected final model is ResNet-18 Full Fine-Tuning because it achieved the best validation accuracy during Phase 4.

## Key Results

- Final Accuracy: 89.7%
- Macro F1-score: 89.59%
- Baseline Accuracy: 67.88%
- Accuracy Improvement: +21.82%
- Total Parameters: 11,242,434
- Checkpoint Size: 42.97 MB
- Single Image Latency: 0.070 seconds
- Batch Size 32 Latency: 1.36 seconds

## Contents

- `notebooks/` contains the final evaluation notebook and HTML export.
- `src/` contains scripts for evaluation, robustness testing, error analysis, latency profiling, and statistical reporting.
- `configs/` contains final evaluation configuration.
- `reports/` contains figures and result visualizations.
- `experiments/` contains CSV results and logs.
- `tests/` contains smoke tests for final evaluation.

## Disclaimer

This project is for academic and educational purposes only. It is not a certified medical diagnosis tool.
