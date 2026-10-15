# Loan Approval Prediction

An educational, end-to-end machine-learning project that compares Logistic Regression, Decision Tree, and Random Forest classifiers on a user-provided binary loan dataset. It saves the best-performing model and provides a Streamlit form for local predictions.

> **Important:** This project is not a credit decision system. Do not use its predictions to approve or reject real applicants. Review applicable fairness, privacy, legal, and regulatory requirements before any real-world use. Protect applicant data and do not publish it to GitHub.

## Project contents

| File | Purpose |
| --- | --- |
| `loan_approval.py` | Dataset validation, preprocessing, stratified model evaluation, report generation, and model saving |
| `app.py` | Streamlit UI that reads the saved model and builds fields from its training schema |
| `tests/test_loan_approval.py` | Focused tests using generated synthetic data |
| `requirements.txt` | Runtime dependencies |

## Setup

Use Python 3.10 or later. From this project folder, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train on your CSV

Keep your CSV local and out of version control. The target must be binary. For the supplied `loan_approval_dataset.csv`, whose target column is `loan_status`, run:

```powershell
python loan_approval.py --data path\to\loan_approval_dataset.csv --target loan_status
```

The pipeline trims whitespace around column names and string values, and recognizes `Y`, `Yes`, `Approved`, `1`, or `True` as the approved class. For another dataset, set the target name explicitly:

```powershell
python loan_approval.py --data path\to\your_dataset.csv --target Loan_Status
```

If your approved class uses a different value, specify it exactly as it appears in the target column:

```powershell
python loan_approval.py --data data\loan_data.csv --target Decision --approved-label Grant
```

The training pipeline excludes `Loan_ID` when present, creates `TotalIncome_log` from `ApplicantIncome` and `CoapplicantIncome` (when both exist), and creates `LoanAmount_log` from `LoanAmount` (when present). It imputes missing values within the training fold, scales numeric fields, one-hot encodes categorical fields, and evaluates all three models on a stratified 20% holdout. Preprocessing is fitted only on the training split. Results are ranked by holdout accuracy, matching the project report's model-selection approach.

### Results on the supplied dataset

The 4,269-row supplied CSV produced the following results using the stratified 20% holdout and fixed random seed of 42:

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 97.5% | 97.9% | 98.1% | 98.0% |
| Decision Tree | 96.8% | 95.5% | 99.6% | 97.5% |
| Logistic Regression | 92.3% | 92.7% | 95.1% | 93.9% |

These are results on this dataset and this single holdout only; they are not evidence of performance on future applicants or real lending decisions.

The command creates:

- `models\loan_model.joblib` — the selected model, preprocessing pipeline, feature schema, and holdout metrics.
- `reports\model_metrics.csv` — accuracy, precision, recall, and F1 for each model.
- `reports\evaluation.png` — holdout accuracy comparison and the selected model's confusion matrix.
- `reports\feature_importance.png` — leading coefficients or tree feature importances for the selected model.

You can override these output locations with `--model-output` and `--reports-dir`.

### Synthetic demo

To exercise the pipeline without a real dataset, explicitly opt into generated synthetic data:

```powershell
python loan_approval.py --demo
```

Synthetic results are for software demonstration only. They are not evidence of real-world model performance and should not be presented as results from actual applicants.

## Run the app

After training:

```powershell
streamlit run app.py
```

For a model artifact at a non-default path, set `LOAN_MODEL_PATH` before launching:

```powershell
$env:LOAN_MODEL_PATH = "C:\path\to\loan_model.joblib"
streamlit run app.py
```

The app uses the training schema stored in the artifact to render inputs and shows the predicted class and its estimated approval probability. These model outputs are not explanations or guarantees.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Data and model safety

Raw data, trained models, and generated reports are ignored by Git by default. Only train on data you are authorized to use. A Joblib artifact can execute code when loaded; load only artifacts you created or otherwise trust.
