# SmartLoan Prediction System

A beginner-friendly BCA mini project built with **Python, Django, Pandas, NumPy, Matplotlib, Seaborn and Scikit-learn**. It demonstrates a complete loan-status classification workflow:

1. Upload a CSV or XLSX dataset.
2. Validate and clean the dataset.
3. View dataset dimensions, missing-value counts and the first 10 rows.
4. Explore EDA charts: distribution, box plot, scatter plot and correlation heatmap.
5. Train a Logistic Regression classifier and view Accuracy, Precision, Recall, F1-score and a Confusion Matrix.
6. Enter a new applicant record and view the model prediction.

> **Educational project only:** The prediction is not an actual bank decision, credit score, or financial recommendation.

## 1. Dataset for the college submission

Use the Kaggle **Loan Prediction / Loan Status** dataset. The commonly used original version has 614 application records and 13 columns, with both numeric and categorical attributes. The target column is `Loan_Status`.

- Kaggle dataset page: https://www.kaggle.com/datasets/ninzaami/loan-predication
- A public dataset description confirms the 13 columns and 614 data records: https://github.com/gagalyash/Loan-status-prediction

Download the training CSV from Kaggle and upload it on the project's **Upload Dataset** page. The source file may be named `loan.csv` or similar; the filename does not matter. The file must contain `Loan_Status`.

Common columns include `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`, and `Loan_Status`. The project automatically removes the identifier column `Loan_ID`.

**Important:** `data/demo_loan_data.csv` is synthetic sample data provided only to test the application. Do not describe it as real-world Kaggle data in your report. For your final college work, download and use the Kaggle file above.

More dataset details and the download steps are in `data/README_dataset.md`.

## 2. Requirements

- Windows 10 or Windows 11
- Python **3.12** recommended
- VS Code (recommended editor)
- Internet connection for installing packages and downloading the Kaggle dataset

Python 3.15 is not recommended for this project because scientific Python packages may not yet publish compatible builds for every new Python release.

## 3. Open the project

1. Extract the ZIP file.
2. Open VS Code.
3. Select **File → Open Folder**.
4. Choose the extracted `SmartLoanPrediction` folder (the folder that contains `manage.py`).
5. Open **Terminal → New Terminal**.

## 4. Create and activate a virtual environment

In the VS Code PowerShell terminal, run these commands one by one:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, run this command in the same terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 5. Install packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Create the local database and start the website

```powershell
python manage.py migrate
python manage.py check
python manage.py runserver
```

Open this address in your browser:

```text
http://127.0.0.1:8000/
```

Keep the terminal open while using the website. Press `Ctrl + C` in the terminal to stop the server.

## 7. Upload and train the model

1. Click **Upload Dataset** in the navigation bar.
2. Select the Kaggle CSV or an XLSX version of the same data.
3. Click **Upload & Train Model**.
4. Wait for the success message. The application will open the Data Summary page.
5. Open **Analytics** to view charts and model metrics.
6. Open **Predict Loan** to enter a new applicant's information.

### Required target column

The target column must be named `Loan_Status` (capitalization is ignored). The app accepts these binary values:

| Approved | Rejected |
|---|---|
| `Y` / `Yes` / `Approved` / `1` | `N` / `No` / `Rejected` / `0` |

If your target has a different name, rename it to `Loan_Status` in Excel or your CSV before uploading. The target must contain both classes.

### Upload rules

- CSV or XLSX only.
- Maximum file size: 10 MB.
- At least 500 records after removing duplicate rows and invalid target rows.
- At least one numeric feature and one categorical feature.
- Keep the target column `Loan_Status` in the file.
- The test/demo file in `data/` has 600 synthetic rows and is not the required real-world dataset.

## 8. What preprocessing does the project perform?

| College requirement | Implementation |
|---|---|
| Missing-value handling | Median imputation for numeric columns; most-frequent imputation for categorical columns |
| Duplicate removal | Removes repeated full rows before training |
| Irrelevant fields | Removes common ID fields such as `Loan_ID`; also removes empty and constant feature columns |
| Outlier treatment | IQR clipping, with limits learned only from training data |
| Categorical encoding | One-Hot Encoding; unknown categories are safely ignored |
| Feature scaling | StandardScaler on numeric columns |
| Feature selection | SelectPercentile with ANOVA F-test, selecting the strongest 80% of encoded features |
| Model | Logistic Regression classifier |
| Evaluation | Accuracy, Precision, Recall, F1-score and Confusion Matrix on a held-out 20% test set |

The preprocessing steps are inside a Scikit-learn Pipeline. This means the imputers, scaling, outlier limits, encoding and feature selection are fitted on the training split, then reused for test data and new predictions. This helps avoid data leakage.

## 9. Project folder structure

```text
SmartLoanPrediction/
│
├── manage.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── SmartLoanPrediction/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── dashboard/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── urls.py
│   ├── views.py
│   ├── ml_service.py
│   ├── templatetags/
│   │   ├── __init__.py
│   │   └── dict_extras.py
│   ├── templates/dashboard/
│   │   ├── base.html
│   │   ├── home.html
│   │   ├── upload.html
│   │   ├── analysis.html
│   │   ├── analytics.html
│   │   └── predict.html
│   └── static/dashboard/style.css
│
├── data/
│   ├── demo_loan_data.csv
│   └── README_dataset.md
│
└── media/
    └── .gitkeep
```

`media/` is created/used at runtime for the active dataset, trained model and metadata. It is ignored by Git because these are generated local files.

## 10. Common errors and fixes

**Error: `py` is not recognized**

Install Python 3.12 and select the option to add Python to PATH. Reopen VS Code after installation.

**Error: `No module named django`**

Activate the virtual environment and run `pip install -r requirements.txt` again.

**Error: `{% extends ... %} must be the first tag`**

In every child template, the `{% extends 'dashboard/base.html' %}` line must be the first Django template tag. In `analysis.html`, it is line 1 and `{% load dict_extras %}` is line 2.

**Error: `Loan_Status` was not found**

Rename the approval/result column to exactly `Loan_Status`.

**Error: fewer than 500 rows**

Use the full Kaggle training dataset, not a small sample or the test-only file.

**Error: XLSX cannot be read**

Install the Excel reader dependency with `pip install openpyxl` and upload an `.xlsx` workbook.

**Error after uploading a new dataset**

Read the error message on the page. The previous trained model is not intentionally replaced unless the new dataset completes training and saving successfully.

## 11. Reset the active model

Stop the server. Delete these generated files from `media/` if they exist:

- `active_dataset.csv`
- `loan_model.joblib`
- `model_metadata.json`

Keep `media/.gitkeep`. Start the server again and upload a dataset to train a new model.
