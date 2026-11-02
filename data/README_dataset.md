# Real Kaggle dataset instructions

For your final college submission, download the **Loan Prediction** dataset from Kaggle:

https://www.kaggle.com/datasets/ninzaami/loan-predication

The commonly used original dataset has 614 application records (plus the CSV header) and 13 columns. It contains both numeric and categorical fields and the target column `Loan_Status`.

## Download steps

1. Open the Kaggle dataset page and sign in to your Kaggle account.
2. Download the dataset ZIP.
3. Extract the ZIP on your computer.
4. Find the CSV training file containing the `Loan_Status` column. It may be named `loan.csv` or have another filename.
5. In SmartLoan, open **Upload Dataset** and select that CSV.

Do not upload a test-only file that does not contain `Loan_Status`.

## Target column expected by this project

`Loan_Status` is the value the model learns to predict:

- `Y` = approved
- `N` = rejected

The application also accepts `Yes/No`, `Approved/Rejected`, and `1/0`. If the target has a different column name, rename it to `Loan_Status` before uploading.

## Included test file

`demo_loan_data.csv` is a generated 600-row synthetic sample for checking that the website flow works. It is not the original Kaggle data and must not be described as real data in your report.
