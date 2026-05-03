# F1 Pit Stop Prediction: Experiment Log

This file tracks the iterative progress of our 10-step experiment strategy.

| Exp # | Description | Key Changes | Target (AUC) | Runtime |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Baseline Pipeline | Initial MLOps pipeline with single LightGBM model. | TBD | TBD |
| **2** | Advanced FE | Ported high-value features (Degradation derivatives, PitWindowPressure). | - | - |
| **3** | High Cardinality | Driver Target Encoding & 2023 Anomaly flag. | - | - |
| **4** | Efficiency | Pruning redundant features via Correlation analysis. | - | - |
| **5** | XGBoost Tuning | Integration of XGBoost (GPU) + Optuna tuning. | - | - |
| **6** | CatBoost Tuning | Integration of CatBoost (GPU) + Optuna tuning. | - | - |
| **7** | Model Diversity | Replaced Deep Learning with lightweight Logistic Regression. | - | - |
| **8** | Validation Strategy | Switched to StratifiedGroupKFold (Race group). | - | - |
| **9** | Fast Ensembling | Replaced Stacking with Nelder-Mead Weighted Blending. | - | - |
| **10** | The Ultimate Run | Final blend using best configs, seeds, and weights. | - | - |

## Summary of Results
*To be filled after each experiment execution.*
