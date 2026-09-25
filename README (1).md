# 🏦 Loan Status Prediction using SVM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning project that predicts whether a loan application will be **approved (Y)** or **rejected (N)** using a **Support Vector Machine (SVM)** classifier.

---

## 📊 Project Overview

| Item | Details |
|------|---------|
| **Algorithm** | Support Vector Machine (Linear Kernel) |
| **Dataset** | Loan Prediction Dataset (614 samples) |
| **Features** | 11 (after preprocessing) |
| **Train/Test Split** | 90% / 10% (stratified) |
| **Training Accuracy** | ~79.86% |
| **Test Accuracy** | ~83.33% |

### Features Used
| Feature | Description |
|---------|-------------|
| `Gender` | Male / Female |
| `Married` | Yes / No |
| `Dependents` | Number of dependents (0, 1, 2, 3+) |
| `Education` | Graduate / Not Graduate |
| `Self_Employed` | Yes / No |
| `ApplicantIncome` | Applicant's income |
| `CoapplicantIncome` | Co-applicant's income |
| `LoanAmount` | Loan amount in thousands |
| `Loan_Amount_Term` | Term of loan in months |
| `Credit_History` | Credit history meets guidelines (1/0) |
| `Property_Area` | Rural / Semiurban / Urban |

**Target:** `Loan_Status` – Y (Approved) or N (Rejected)

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/loan-status-prediction.git
cd loan-status-prediction
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebook
```bash
jupyter notebook Project_6_Loan_Status_Prediction.ipynb
```

Or run the Python script:
```bash
python loan_status_prediction.py
```

> **Note:** The notebook will automatically download the dataset if `dataset.csv` is not found locally.

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Training Accuracy | 79.86% |
| Test Accuracy | 83.33% |

The model performs well on unseen data, with test accuracy slightly higher than training accuracy, indicating good generalization without overfitting.

---

## 📁 Project Structure

```
loan-status-prediction/
│
├── Project_6_Loan_Status_Prediction.ipynb   # Main notebook
├── loan_status_prediction.py                # Python script version
├── requirements.txt                         # Dependencies
├── .gitignore                               # Git ignore rules
├── LICENSE                                  # MIT License
└── README.md                                # Documentation
```

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **NumPy** & **Pandas** – Data handling
- **Seaborn** – Visualization
- **Scikit-learn** – SVM model, train/test split, accuracy metrics

---

## 📝 Preprocessing Steps

1. Handle missing values (drop null rows)
2. Replace `3+` dependents with `4`
3. Encode categorical variables to numerical:
   - Gender: Male=1, Female=0
   - Married: Yes=1, No=0
   - Education: Graduate=1, Not Graduate=0
   - Self_Employed: Yes=1, No=0
   - Property_Area: Rural=0, Semiurban=1, Urban=2
   - Loan_Status: Y=1, N=0
4. Separate features (X) and target (Y)
5. Stratified train-test split (90/10)

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

Feel free to fork, star ⭐, and contribute!

---

**Made with ❤️ for learning Machine Learning**
