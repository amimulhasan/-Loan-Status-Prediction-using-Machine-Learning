"""
Loan Status Prediction using Support Vector Machine (SVM)
=========================================================
Predicts whether a loan application will be approved or rejected.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")


def load_dataset():
    """Load loan dataset from local file or download if not found."""
    if os.path.exists("dataset.csv"):
        print("Loading dataset from dataset.csv ...")
        return pd.read_csv("dataset.csv")
    elif os.path.exists("/content/dataset.csv"):
        print("Loading dataset from /content/dataset.csv ...")
        return pd.read_csv("/content/dataset.csv")
    else:
        print("dataset.csv not found. Downloading from public source ...")
        url = "https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv"
        df = pd.read_csv(url)
        df.to_csv("dataset.csv", index=False)
        print("Dataset downloaded and saved as dataset.csv")
        return df


def preprocess(df):
    """Clean and encode the dataset."""
    print("\n--- Preprocessing ---")
    print(f"Original shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")

    # Drop rows with missing values
    df = df.dropna()
    print(f"Shape after dropping nulls: {df.shape}")

    # Encode Loan_Status
    df.replace({"Loan_Status": {"N": 0, "Y": 1}}, inplace=True)

    # Replace 3+ dependents with 4
    df = df.replace(to_replace="3+", value=4)

    # Encode categorical columns
    df.replace(
        {
            "Married": {"No": 0, "Yes": 1},
            "Gender": {"Male": 1, "Female": 0},
            "Self_Employed": {"No": 0, "Yes": 1},
            "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
            "Education": {"Graduate": 1, "Not Graduate": 0},
        },
        inplace=True,
    )

    # Convert Dependents to numeric
    df["Dependents"] = df["Dependents"].astype(int)

    print("Categorical columns encoded to numerical values.")
    return df


def main():
    print("=" * 55)
    print("  Loan Status Prediction using SVM")
    print("=" * 55)

    # Load data
    print("\n[1/5] Loading dataset...")
    loan_dataset = load_dataset()
    print(f"Dataset shape: {loan_dataset.shape}")
    print(f"\nFirst 5 rows:\n{loan_dataset.head()}")

    # Preprocess
    print("\n[2/5] Preprocessing data...")
    loan_dataset = preprocess(loan_dataset)

    # Visualize Education vs Loan Status
    print("\n[3/5] Generating visualizations...")
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Education", hue="Loan_Status", data=loan_dataset)
    plt.title("Education vs Loan Status")
    plt.tight_layout()
    plt.savefig("education_vs_loan_status.png", dpi=150)
    print("Saved: education_vs_loan_status.png")

    plt.figure(figsize=(8, 5))
    sns.countplot(x="Married", hue="Loan_Status", data=loan_dataset)
    plt.title("Marital Status vs Loan Status")
    plt.tight_layout()
    plt.savefig("married_vs_loan_status.png", dpi=150)
    print("Saved: married_vs_loan_status.png")

    # Separate features and target
    X = loan_dataset.drop(columns=["Loan_ID", "Loan_Status"], axis=1)
    Y = loan_dataset["Loan_Status"]

    # Train-test split
    print("\n[4/5] Splitting data (90% train / 10% test)...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, stratify=Y, random_state=2
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Train SVM
    print("\n[5/5] Training SVM (linear kernel)...")
    classifier = svm.SVC(kernel="linear")
    classifier.fit(X_train, Y_train)
    print("Model trained successfully!")

    # Evaluate
    X_train_pred = classifier.predict(X_train)
    train_acc = accuracy_score(X_train_pred, Y_train)

    X_test_pred = classifier.predict(X_test)
    test_acc = accuracy_score(X_test_pred, Y_test)

    print("\n--- Results ---")
    print(f"Accuracy on training data : {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Accuracy on test data     : {test_acc:.4f} ({test_acc*100:.2f}%)")

    print("\nClassification Report (Test Set):")
    print(classification_report(Y_test, X_test_pred, target_names=["Rejected (N)", "Approved (Y)"]))

    # Confusion matrix
    cm = confusion_matrix(Y_test, X_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved: confusion_matrix.png")

    print("\n" + "=" * 55)
    print("Done! Check the generated plots.")
    print("=" * 55)


if __name__ == "__main__":
    main()
