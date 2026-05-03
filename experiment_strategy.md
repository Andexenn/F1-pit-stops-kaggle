# F1 Pit Stop Prediction: Experiment Strategy

## Objective
Combine the robust feature engineering of the high-scoring notebook (`@pit-or-stay-f1-strategy-1.ipynb`) with the runtime efficiency of the fast notebook (`@sebastian-vettel-aaaa.ipynb`). The pipeline will strictly follow the architecture outlined in `pipeline.png` to ensure reproducibility and clean code structure. A key focus will be **progressive data visualization** to understand the impact of our changes.

## The 10-Step Experiment Plan

### Phase 1: Foundation & Pipeline Setup
*   **Experiment 1: Baseline Pipeline Execution & Initial EDA**
    *   **Goal:** Establish the MLOps pipeline and understand the raw data.
    *   **Visualization:** Plot the target class distribution (Pit vs. No Pit) and basic feature distributions (e.g., TyreLife).
    *   **Action:** Train a fast, single LightGBM model on GPU using raw numeric features and label-encoded categorical features. 
*   **Experiment 2: Advanced Feature Engineering**
    *   **Goal:** Integrate the most powerful features from the slow notebook.
    *   **Visualization:** Create scatter plots or box plots showing the relationship between new features (like `Cumulative_Degradation` derivatives) and the target.
    *   **Action:** Add features like `TyreLifeBin`, `Cumulative_Degradation` derivatives (e.g., `Deg_rolling_diff`), and `PitWindowPressure` (TyreLife × RaceProgress).

### Phase 2: Handling High Cardinality & Anomalies
*   **Experiment 3: Target Encoding & The 2023 Anomaly**
    *   **Goal:** Address the 2023 season anomaly and extract signal from high-cardinality categorical features.
    *   **Visualization:** Plot Pit Rate by Year (to highlight the 2023 anomaly) and a bar chart of top drivers by pit frequency before and after target encoding.
    *   **Action:** Introduce Out-of-Fold (OOF) Target Encoding for the `Driver` feature. Add an `is_2023` binary flag.

### Phase 3: Efficiency & Pruning
*   **Experiment 4: Feature Selection & Pruning**
    *   **Goal:** Decrease runtime by removing redundant data.
    *   **Visualization:** Generate a **Correlation Heatmap** to identify highly correlated features. Plot Feature Importance from the LightGBM model.
    *   **Action:** Drop highly correlated features (e.g., redundant lap counts vs. race progress) based on the heatmap and importance scores.

### Phase 4: Model Diversity & Tuning
*   **Experiment 5: XGBoost Integration & Optuna Tuning**
    *   **Goal:** Bring in the second fast GPU model.
    *   **Visualization:** Plot Optuna optimization history (e.g., parallel coordinate plot of hyperparameter relationships).
    *   **Action:** Train XGBoost on the pruned feature set. Use Optuna to find the optimal hyperparameters quickly.
*   **Experiment 6: CatBoost Integration & Optuna Tuning**
    *   **Goal:** Bring in CatBoost to handle categorical features natively.
    *   **Visualization:** Compare Feature Importance across LGBM, XGBoost, and CatBoost using a grouped bar chart.
    *   **Action:** Train CatBoost (GPU) utilizing its native categorical handling. Tune with Optuna.
*   **Experiment 7: Deep Learning Drop / Lightweight Replacement**
    *   **Goal:** Maintain diversity without the massive 1-hour penalty of the PyTorch MLP.
    *   **Visualization:** Plot a ROC curve comparing the lightweight model to the gradient boosting models.
    *   **Action:** Replace the Deep Learning component with a lightweight Logistic Regression or a highly regularized Random Forest on scaled features.

### Phase 5: Validation & Ensembling
*   **Experiment 8: Advanced Cross-Validation Strategy**
    *   **Goal:** Ensure the model generalizes perfectly to unseen data.
    *   **Visualization:** Plot CV score variance across different folds to show stability improvements.
    *   **Action:** Switch from standard StratifiedKFold to `GroupKFold` or `StratifiedGroupKFold` grouped by `Race` to prevent data leakage from consecutive laps.
*   **Experiment 9: Fast Ensembling (Replacing Heavy Stacking)**
    *   **Goal:** Achieve the score bump of stacking without the long runtime.
    *   **Visualization:** Create a bar chart showing the final assigned weights for each model in the ensemble.
    *   **Action:** Use a fast Weighted Average (e.g., Nelder-Mead optimization) to find the optimal blend weights for the models instead of multi-layer stacking.

### Phase 6: The Ultimate Run
*   **Experiment 10: The Best of the Best**
    *   **Goal:** Maximum Score, Minimum Runtime.
    *   **Visualization:** Final ROC-AUC curve and a Confusion Matrix for the blended predictions.
    *   **Action:** Run the final pipeline using the pruned dataset, optimal Optuna configurations, and the fast Nelder-Mead ensemble weights.

---

## Experiment 1 Notebook Blueprint
The first notebook will implement the visual pipeline from `pipeline.png` using modular Python functions. **Crucially, every major step will be preceded by a descriptive Markdown cell explaining the action being taken in the context of the pipeline.**

1.  **Markdown:** `## 1. Raw Data` (Explanation of loading procedures)
    *   Load CSVs.
2.  **Markdown:** `## 2. Data Validation` (Explanation of quality checks)
    *   Check for NaNs, extreme outliers, and correct data types.
3.  **Markdown:** `## 3. Exploratory Data Analysis (EDA)` (Visualizing the raw state)
    *   Plot Target Distribution and basic feature distributions.
4.  **Markdown:** `## 4. Data Preprocessing` (Explanation of cleaning and encoding)
    *   Clean column names.
    *   Ordinal Encode `Driver`, `Compound`, `Race`.
5.  **Markdown:** `## 5. Model Training` (Explanation of the baseline model choice)
    *   Train a baseline LightGBM model (`device='gpu'`).
6.  **Markdown:** `## 6. Model Evaluation` (Explanation of the metric)
    *   Calculate ROC-AUC on a simple train/test split.
7.  **Markdown:** `## 7. Model Validation` (Explanation of Cross-Validation)
    *   Perform a 5-Fold Stratified Cross-Validation to ensure stability.
8.  **Markdown:** `## 8. Model Deployment & Feedback` (Explanation of final outputs)
    *   Save the model artifact (`baseline_lgbm.pkl`) and output a submission file.
