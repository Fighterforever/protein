# Protein Corona Prediction

This project provides a machine-learning pipeline for predicting the protein corona profiles of nanoparticles from experimental metadata. The main model trains on tabular features and protein annotations to rank the most likely proteins binding to each sample.

## Data
- **imputed.xlsx** – experimental metadata and imputed protein intensity matrix.
- **imputed_prot_ann.csv** – annotation table with numeric features per protein (e.g. physico‑chemical properties).

Both files are included in the repository for demonstration purposes.

## Installation
Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn xgboost shap joblib
```

## Running an experiment
Execute the training and evaluation script:

```bash
python pc_run.py --data_xlsx imputed.xlsx \
                 --prot_csv imputed_prot_ann.csv \
                 --outdir results
```

The command will write metrics, predictions and feature importance files into the specified `results/` directory.

## Main scripts
- **pc_run.py** – end‑to‑end pipeline that builds folds, trains models (ridge regression, XGBRanker and mixture‑of‑experts) and exports evaluation reports.
- **Data_Exploration.ipynb** – notebook for inspecting the dataset and producing additional plots.

## Outputs
Running `pc_run.py` produces:

| File | Description |
| --- | --- |
| `metrics_summary.csv` | Mean performance over folds. |
| `metrics_by_group.csv` | Metrics grouped by predefined protein classes. |
| `feature_importance_global.csv` | Top features from the global ranker. |
| `feature_importance_shap.csv` | SHAP values for detailed feature attribution. |
| `prediction_examples.csv` | Examples of predicted vs. true top proteins. |
| `README_results.md` | Text summary of the experiment. |

## Visualisation
The SHAP output can be visualised with the [SHAP library](https://shap.readthedocs.io):

```python
import pandas as pd
import shap

data = pd.read_csv('results/feature_importance_shap.csv')
shap.summary_plot(data.drop(columns=['feature']).values, feature_names=data['feature'])
```

Adjust the code above to point to your results directory.


