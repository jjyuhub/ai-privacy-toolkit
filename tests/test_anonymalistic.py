# Standard library imports
import time

# Scientific computing
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import ttest_rel, f_oneway

# Machine Learning - Scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ML Models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Deep Learning - PyTorch
from torch import nn, optim, sigmoid, where, from_numpy
from torch.nn import functional

# Testing
import pytest

# Dataset utilities
from sklearn.datasets import load_diabetes

# Privacy tools (apt)
from apt.anonymization import Anonymize
from apt.utils.datasets.datasets import PytorchData
from apt.utils.datasets import ArrayDataset
from apt.utils.dataset_utils import (
    get_iris_dataset_np,
    get_adult_dataset_pd,
    get_nursery_dataset_pd
)
from apt.utils.models import CLASSIFIER_MULTI_OUTPUT_BINARY_LOGITS
from apt.utils.models.pytorch_model import PyTorchClassifier

def test_feature_importance_shift_multiple_k():
    """
    Compares feature importance shifts for k-anonymization with k=5, k=10, and k=100.
    Ensures we observe the trade-off between privacy and model utility.
    """

    print("\n===== STARTING TEST: Feature Importance Shift Across k-Values =====\n")

    # Step 1: Load Dataset
    print("[Step 1] Loading the Adult dataset...")
    (x_train, y_train), _ = get_adult_dataset_pd()
    feature_names = x_train.columns.tolist()

    # Identify categorical and numerical features
    categorical_features = x_train.select_dtypes(include=['object']).columns.tolist()
    numerical_features = x_train.select_dtypes(exclude=['object']).columns.tolist()

    print(f" - Categorical Features: {categorical_features}")
    print(f" - Numerical Features: {numerical_features}\n")

    # Step 2: Preprocess Data (One-Hot Encoding)
    print("[Step 2] Applying One-Hot Encoding to categorical features...")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
        ]
    )

    # Apply transformation
    x_train_encoded = preprocessor.fit_transform(x_train)

    # Get updated feature names after encoding
    encoded_feature_names = (
        numerical_features +
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    )

    print(f" - Transformed dataset has {x_train_encoded.shape[1]} features after encoding.\n")

    # Step 3: Train Initial Decision Tree
    print("[Step 3] Training Decision Tree Classifier on the original dataset...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(x_train_encoded, y_train)

    # Extract feature importance before anonymization
    feature_importance_before = model.feature_importances_

    importance_df_before = pd.DataFrame({
        'Feature': encoded_feature_names,
        'Importance': feature_importance_before
    }).sort_values(by="Importance", ascending=False)

    print("\nTop 10 Features Before Anonymization:")
    print(importance_df_before.head(10).to_string(index=False))

    # Define k-values to test
    k_values = [5, 10, 100]

    # Iterate over different k-values
    for k in k_values:
        print(f"\n===== Testing k-Anonymization with k={k} =====\n")

        # Step 4: Apply k-Anonymization
        print(f"[Step 4] Applying k={k} Anonymization on quasi-identifiers...")
        quasi_identifiers_original = ["age", "education-num", "marital-status", "occupation"]

        # Map original quasi-identifiers to encoded versions
        quasi_identifiers_encoded = []
        for qi in quasi_identifiers_original:
            if qi in numerical_features:
                quasi_identifiers_encoded.append(qi)
            else:
                # Find corresponding one-hot encoded features
                encoded_qi_features = [feat for feat in encoded_feature_names if qi in feat]
                quasi_identifiers_encoded.extend(encoded_qi_features)

        print(f" - Selected Quasi-Identifiers after Encoding: {quasi_identifiers_encoded}")
        print(" - Applying anonymization...\n")

        # Convert dataset to DataFrame for consistency
        x_train_encoded_df = pd.DataFrame(x_train_encoded, columns=encoded_feature_names)

        anonymizer = Anonymize(k, quasi_identifiers_encoded, categorical_features=quasi_identifiers_encoded)

        # Apply Anonymization
        anonymized_data = anonymizer.anonymize(ArrayDataset(x_train_encoded_df, y_train, encoded_feature_names))

        # Convert anonymized dataset to numerical format
        anonymized_encoded = np.array(anonymized_data, dtype=np.float64)

        # Step 5: Train Decision Tree on Anonymized Data
        print("[Step 5] Training Decision Tree Classifier on anonymized dataset...")
        model_after = DecisionTreeClassifier(random_state=42)
        model_after.fit(anonymized_encoded, y_train)

        # Extract feature importance after anonymization
        feature_importance_after = model_after.feature_importances_

        importance_df_after = pd.DataFrame({
            'Feature': encoded_feature_names,
            'Importance': feature_importance_after
        }).sort_values(by="Importance", ascending=False)

        print(f"\nTop 10 Features After Anonymization (k={k}):")
        print(importance_df_after.head(10).to_string(index=False))

        # Step 6: Analyze Feature Importance Shift
        print(f"\n[Step 6] Comparing Feature Importance Shift Before and After Anonymization (k={k})...")
        comparison_df = importance_df_before.merge(importance_df_after, on="Feature", suffixes=("_Before", f"_After_k{k}"))
        comparison_df[f"Importance_Change_k{k}"] = comparison_df[f"Importance_After_k{k}"] - comparison_df["Importance_Before"]

        # Sort by absolute importance change
        comparison_df = comparison_df.sort_values(by=f"Importance_Change_k{k}", ascending=True)

        print(f"\nFeatures with the Most Reduction in Importance (k={k}):")
        print(comparison_df.head(10).to_string(index=False))

        print(f"\nFeatures with the Most Increase in Importance (k={k}):")
        print(comparison_df.tail(10).to_string(index=False))

        # Step 7: Conclusion for this k-value
        print(f"\n[Step 7] Drawing Conclusions for k={k}...")
        most_reduced_features = comparison_df.nsmallest(5, f"Importance_Change_k{k}")
        most_important_lost_feature = most_reduced_features.iloc[0]["Feature"]

        print(f"\n✅ The most affected feature for k={k} was: **{most_important_lost_feature}**, losing {most_reduced_features.iloc[0][f'Importance_Change_k{k}']:.4f} importance.")

        if most_reduced_features[f"Importance_Change_k{k}"].min() < -0.05:
            print(f"⚠️ Anonymization (k={k}) significantly reduced the importance of key predictive features.")
        else:
            print(f"✅ Anonymization (k={k}) had minimal impact on feature importance.")

        print(f"\n===== TEST COMPLETE for k={k} =====\n")

def test_model_accuracy_retention():
    """
    Compares model performance before and after k-anonymization at k=5, k=10, k=100.
    Evaluates Accuracy, Precision, Recall, and F1-score across different models.
    """

    print("\n===== STARTING TEST: Model Accuracy Retention Across k-Values =====\n")

    # Step 1: Load Dataset
    print("[Step 1] Loading the Adult dataset...")
    (x_train, y_train), (x_test, y_test) = get_adult_dataset_pd()
    feature_names = x_train.columns.tolist()

    # Identify categorical and numerical features
    categorical_features = x_train.select_dtypes(include=['object']).columns.tolist()
    numerical_features = x_train.select_dtypes(exclude=['object']).columns.tolist()

    print(f" - Categorical Features: {categorical_features}")
    print(f" - Numerical Features: {numerical_features}\n")

    # Step 2: Preprocess Data (One-Hot Encoding)
    print("[Step 2] Applying One-Hot Encoding to categorical features...")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
        ]
    )

    # Apply transformation
    x_train_encoded = preprocessor.fit_transform(x_train)
    x_test_encoded = preprocessor.transform(x_test)

    # Get updated feature names after encoding
    encoded_feature_names = (
        numerical_features +
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    )

    print(f" - Transformed dataset has {x_train_encoded.shape[1]} features after encoding.\n")

    # Define models for testing
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }

    # Train and evaluate models on original (non-anonymized) data
    print("[Step 3] Evaluating models on original dataset...")
    results = {"k-value": [], "Model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1-Score": []}

    for model_name, model in models.items():
        model.fit(x_train_encoded, y_train)
        y_pred = model.predict(x_test_encoded)

        results["k-value"].append("No Anonymization")
        results["Model"].append(model_name)
        results["Accuracy"].append(accuracy_score(y_test, y_pred))
        results["Precision"].append(precision_score(y_test, y_pred, average='weighted'))
        results["Recall"].append(recall_score(y_test, y_pred, average='weighted'))
        results["F1-Score"].append(f1_score(y_test, y_pred, average='weighted'))

    # Define k-values to test
    k_values = [5, 10, 100]

    # Iterate over different k-values
    for k in k_values:
        print(f"\n===== Testing k-Anonymization with k={k} =====\n")

        # Step 4: Apply k-Anonymization
        print(f"[Step 4] Applying k={k} Anonymization on quasi-identifiers...")
        quasi_identifiers_original = ["age", "education-num", "marital-status", "occupation"]

        # Map original quasi-identifiers to encoded versions
        quasi_identifiers_encoded = []
        for qi in quasi_identifiers_original:
            if qi in numerical_features:
                quasi_identifiers_encoded.append(qi)
            else:
                # Find corresponding one-hot encoded features
                encoded_qi_features = [feat for feat in encoded_feature_names if qi in feat]
                quasi_identifiers_encoded.extend(encoded_qi_features)

        print(f" - Selected Quasi-Identifiers after Encoding: {quasi_identifiers_encoded}")
        print(" - Applying anonymization...\n")

        # Convert dataset to DataFrame for consistency
        x_train_encoded_df = pd.DataFrame(x_train_encoded, columns=encoded_feature_names)

        anonymizer = Anonymize(k, quasi_identifiers_encoded, categorical_features=quasi_identifiers_encoded)

        # Apply Anonymization
        anonymized_data = anonymizer.anonymize(ArrayDataset(x_train_encoded_df, y_train, encoded_feature_names))

        # Convert anonymized dataset to numerical format
        anonymized_encoded = np.array(anonymized_data, dtype=np.float64)

        # Step 5: Train and evaluate models on anonymized data
        print("[Step 5] Evaluating models on anonymized dataset...")

        for model_name, model in models.items():
            model.fit(anonymized_encoded, y_train)
            y_pred = model.predict(x_test_encoded)

            results["k-value"].append(f"k={k}")
            results["Model"].append(model_name)
            results["Accuracy"].append(accuracy_score(y_test, y_pred))
            results["Precision"].append(precision_score(y_test, y_pred, average='weighted'))
            results["Recall"].append(recall_score(y_test, y_pred, average='weighted'))
            results["F1-Score"].append(f1_score(y_test, y_pred, average='weighted'))

    # Convert results to DataFrame and print
    results_df = pd.DataFrame(results)
    print("\n===== Final Model Performance Across k-Values =====")
    print(results_df.to_string(index=False))

    # Step 6: Draw conclusions
    print("\n===== Conclusion =====")
    print("🔹 Lower k (e.g., k=5) should retain more accuracy but provide less privacy.")
    print("🔹 Higher k (e.g., k=100) should enforce stronger privacy but may reduce model performance.")
    print("🔹 If utility is critical, k=10 might be an optimal balance.")
    print("\n===== TEST COMPLETE: Model Accuracy Retention Analysis Finished =====\n")





def test_training_time_overhead():
    """
    Compares training time of models before and after anonymization (k=10, k=100).
    Evaluates whether stronger privacy (higher k) introduces significant computational costs.
    """

    print("\n===== STARTING TEST: Training Time Overhead Due to Anonymization =====\n")

    # Step 1: Load Dataset
    print("[Step 1] Loading the Adult dataset...")
    (x_train, y_train), _ = get_adult_dataset_pd()
    feature_names = x_train.columns.tolist()

    # Identify categorical and numerical features
    categorical_features = x_train.select_dtypes(include=['object']).columns.tolist()
    numerical_features = x_train.select_dtypes(exclude=['object']).columns.tolist()

    print(f" - Categorical Features: {categorical_features}")
    print(f" - Numerical Features: {numerical_features}\n")

    # Step 2: Measure Feature Encoding Time
    print("[Step 2] Applying One-Hot Encoding to categorical features...")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
        ]
    )

    start_encoding_time = time.time()
    x_train_encoded = preprocessor.fit_transform(x_train)
    encoding_time = time.time() - start_encoding_time  # Time taken for feature encoding

    # Get updated feature names after encoding
    encoded_feature_names = (
        numerical_features +
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    )

    print(f" - Transformed dataset has {x_train_encoded.shape[1]} features after encoding.")
    print(f"✅ Feature encoding completed in {encoding_time:.4f} seconds.\n")

    # Define models for testing
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }

    # Define k-values to test
    k_values = [10, 100]

    # Store results
    results = {
        "k-value": [], "Model": [], "Feature Encoding Time (sec)": [], 
        "Anonymization Time (sec)": [], "Training Time (sec)": []
    }

    # Step 3: Measure Training Time on Original Data
    print("[Step 3] Measuring training time on original dataset...")

    for model_name, model in models.items():
        start_time = time.time()
        model.fit(x_train_encoded, y_train)
        training_time = time.time() - start_time

        results["k-value"].append("No Anonymization")
        results["Model"].append(model_name)
        results["Feature Encoding Time (sec)"].append(encoding_time)
        results["Anonymization Time (sec)"].append(0.0)  # No anonymization for baseline
        results["Training Time (sec)"].append(training_time)

        print(f"✅ {model_name} trained in {training_time:.4f} seconds (No Anonymization).")

    # Iterate over different k-values
    for k in k_values:
        print(f"\n===== Testing k-Anonymization with k={k} =====\n")

        # Step 4: Apply k-Anonymization and Measure Time
        print(f"[Step 4] Applying k={k} Anonymization on quasi-identifiers...")
        quasi_identifiers_original = ["age", "education-num", "marital-status", "occupation"]

        # Map original quasi-identifiers to encoded versions
        quasi_identifiers_encoded = []
        for qi in quasi_identifiers_original:
            if qi in numerical_features:
                quasi_identifiers_encoded.append(qi)
            else:
                # Find corresponding one-hot encoded features
                encoded_qi_features = [feat for feat in encoded_feature_names if qi in feat]
                quasi_identifiers_encoded.extend(encoded_qi_features)

        print(f" - Selected Quasi-Identifiers after Encoding: {quasi_identifiers_encoded}")
        print(" - Applying anonymization...\n")

        # Convert dataset to DataFrame for consistency
        x_train_encoded_df = pd.DataFrame(x_train_encoded, columns=encoded_feature_names)

        anonymizer = Anonymize(k, quasi_identifiers_encoded, categorical_features=quasi_identifiers_encoded)

        # Measure Anonymization Time
        start_anonymization_time = time.time()
        anonymized_data = anonymizer.anonymize(ArrayDataset(x_train_encoded_df, y_train, encoded_feature_names))
        anonymization_time = time.time() - start_anonymization_time  # Time taken for anonymization

        print(f"✅ Anonymization for k={k} completed in {anonymization_time:.4f} seconds.")

        # Convert anonymized dataset to numerical format
        anonymized_encoded = np.array(anonymized_data, dtype=np.float64)

        # Step 5: Measure Training Time on Anonymized Data
        print("[Step 5] Measuring training time on anonymized dataset...")

        for model_name, model in models.items():
            start_time = time.time()
            model.fit(anonymized_encoded, y_train)
            training_time = time.time() - start_time

            results["k-value"].append(f"k={k}")
            results["Model"].append(model_name)
            results["Feature Encoding Time (sec)"].append(encoding_time)
            results["Anonymization Time (sec)"].append(anonymization_time)
            results["Training Time (sec)"].append(training_time)

            print(f"✅ {model_name} trained in {training_time:.4f} seconds (k={k}).")

    # Convert results to DataFrame and print
    results_df = pd.DataFrame(results)
    print("\n===== Final Training Time Across k-Values =====")
    print(results_df.to_string(index=False))

    # Step 6: Draw conclusions
    print("\n===== Conclusion =====")
    print("🔹 Lower k (e.g., k=10) should retain reasonable training times.")
    print("🔹 Higher k (e.g., k=100) may significantly increase preprocessing and training overhead.")
    print("🔹 If efficiency is crucial, choosing a moderate k-value like k=10 is recommended.")
    print("\n===== TEST COMPLETE: Training Time Overhead Analysis Finished =====\n")






