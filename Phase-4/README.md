# Phase 4 - Transfer Learning, Autoencoder and Hyperparameter Optimization

This folder contains Phase 4 work for DermaVision AI.

## Phase 4 Focus

- Transfer learning using ResNet-18
- Feature extraction experiment
- Full fine-tuning experiment
- Differential learning rate experiment
- Autoencoder / VAE component
- Hyperparameter optimization
- Model comparison and analysis

## Contents

- `notebooks/` contains Phase 4 experiment notebooks and HTML exports.
- `src/` contains training scripts for transfer learning, autoencoder, HPO analysis, and evaluation.
- `configs/` contains YAML configuration files for each experiment.
- `reports/` contains training curves, reconstruction images, latent space plots, and comparison figures.
- `experiments/` contains CSV logs and HPO results.
- `tests/` contains smoke tests for Phase 4 models.

## Transfer Learning Experiments

1. Feature Extraction
2. Full Fine-Tuning
3. Differential Learning Rate

## Generative Component

An autoencoder is used to learn compressed latent representations of skin lesion images and reconstruct input images.

## Hyperparameter Optimization

Bayesian hyperparameter optimization is planned using Optuna-style search over learning rate, dropout, batch size, optimizer, and weight decay.

## Disclaimer

This project is for academic and educational purposes only. It is not a certified medical diagnosis tool.
