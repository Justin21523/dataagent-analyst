# R4: Model Explainability

## Goal

Add global and local model explanations to the ML Workbench.

## Diagnostics

### Classification

- Confusion matrix
- One-vs-rest ROC curves
- One-vs-rest precision-recall curves
- ROC AUC
- Average precision
- Binary calibration curve
- Misclassified samples

### Regression

- Predicted versus actual
- Residual plot
- Mean residual
- Residual standard deviation
- Median residual
- Maximum absolute error
- Highest-error samples

## Global Explainability

### Permutation Importance

Permutation importance is computed on raw holdout features against the full scikit-learn Pipeline.

This explains the effect of:

```text
Raw feature
→ Preprocessing
→ Model
→ Evaluation score
