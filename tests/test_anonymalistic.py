import pytest  # Pytest for unit testing
import numpy as np  # NumPy for numerical computations
import pandas as pd  # Pandas for data manipulation
from sklearn.compose import ColumnTransformer  # Helps apply different preprocessing techniques to different columns
from sklearn.impute import SimpleImputer  # Handles missing values
from sklearn.pipeline import Pipeline  # For creating machine learning workflows
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # Decision tree models for classification and regression
from sklearn.preprocessing import OneHotEncoder  # Converts categorical features into numerical format
from sklearn.datasets import load_diabetes  # Loads the diabetes dataset for regression testing
from sklearn.model_selection import train_test_split  # Splits data into training and testing sets
from torch import nn, optim, sigmoid, where, from_numpy  # PyTorch for deep learning models
from torch.nn import functional  # PyTorch functional utilities
from scipy.special import expit  # Sigmoid function for probability conversions

# Import utilities from the `apt` library (Anonymization & Privacy Tools)
from apt.utils.datasets.datasets import PytorchData  # Wrapper for PyTorch dataset
from apt.utils.models import CLASSIFIER_MULTI_OUTPUT_BINARY_LOGITS  # Classification mode definition for multi-label binary classification
from apt.utils.models.pytorch_model import PyTorchClassifier  # Wrapper for PyTorch-based classifiers
from apt.anonymization import Anonymize  # Anonymization utility for generalizing data
from apt.utils.dataset_utils import get_iris_dataset_np, get_adult_dataset_pd, get_nursery_dataset_pd  # Load datasets
from apt.utils.datasets import ArrayDataset  # Wrapper for structured dataset representation

import numpy as np
from apt.anonymization import Anonymize
from apt.utils.dataset_utils import get_iris_dataset_np
from apt.utils.datasets import ArrayDataset
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from apt.anonymization import Anonymize
from apt.utils.dataset_utils import get_adult_dataset_pd
from apt.utils.datasets import ArrayDataset

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from apt.anonymization import Anonymize
from apt.utils.dataset_utils import get_adult_dataset_pd
from apt.utils.datasets import ArrayDataset

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from apt.anonymization import Anonymize
from apt.utils.dataset_utils import get_adult_dataset_pd
from apt.utils.datasets import ArrayDataset

def test_feature_importance_shift():
    """
    Evaluates how k-anonymization affects feature importance in a Decision Tree model.
    Fixes the issue by encoding categorical features before training AND before anonymization.
    """

    print("\n===== STARTING TEST: Feature Importance Shift Due to Anonymization =====\n")

    # Step 1: Load Dataset
    print("[Step 1] Loading the Adult dataset...")
    (x_train, y_train), _ = get_adult_dataset_pd()
    feature_names = x_train.columns.tolist()

    print(f" - Number of records: {x_train.shape[0]}")
    print(f" - Number of features: {len(feature_names)}")
    print(f" - Feature Names: {feature_names}\n")

    # Identify categorical and numerical features
    categorical_features = x_train.select_dtypes(include=['object']).columns.tolist()
    numerical_features = x_train.select_dtypes(exclude=['object']).columns.tolist()

    print(f" - Categorical Features: {categorical_features}")
    print(f" - Numerical Features: {numerical_features}\n")

    # Step 2: Preprocess Data (One-Hot Encode Categorical Features)
    print("[Step 2] Applying One-Hot Encoding to categorical features...")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),  # Keep numerical features as-is
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)  # One-hot encode categorical features
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

    # Step 4: Apply k-Anonymization (k=10)
    print("\n[Step 4] Applying k-Anonymization (k=10) on quasi-identifiers...")
    k = 10  # Minimum number of individuals per group
    quasi_identifiers = ["age", "education-num", "marital-status", "occupation"]

    print(f" - Selected Quasi-Identifiers: {quasi_identifiers}")
    print(" - Applying anonymization...\n")

    # Apply the SAME transformation to the anonymized dataset BEFORE feeding into `Anonymize`
    x_train_encoded_df = pd.DataFrame(x_train_encoded, columns=encoded_feature_names)

    anonymizer = Anonymize(k, quasi_identifiers, categorical_features=quasi_identifiers)

    # Convert anonymized dataset to numerical format using pre-trained encoder
    anonymized_data = anonymizer.anonymize(ArrayDataset(x_train_encoded_df, y_train, encoded_feature_names))

    # Apply same transformation to anonymized data (ensuring feature consistency)
    anonymized_encoded = np.array(anonymized_data, dtype=np.float64)  # Convert back to numerical format

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

    print("\nTop 10 Features After Anonymization:")
    print(importance_df_after.head(10).to_string(index=False))

    # Step 6: Analyze Feature Importance Shift
    print("\n[Step 6] Comparing Feature Importance Shift Before and After Anonymization...")
    comparison_df = importance_df_before.merge(importance_df_after, on="Feature", suffixes=("_Before", "_After"))
    comparison_df["Importance_Change"] = comparison_df["Importance_After"] - comparison_df["Importance_Before"]

    # Sort by absolute importance change
    comparison_df = comparison_df.sort_values(by="Importance_Change", ascending=True)

    print("\nFeatures with the Most Reduction in Importance:")
    print(comparison_df.head(10).to_string(index=False))

    print("\nFeatures with the Most Increase in Importance:")
    print(comparison_df.tail(10).to_string(index=False))

    # Step 7: Conclusion
    print("\n[Step 7] Drawing Conclusions...")
    most_reduced_features = comparison_df.nsmallest(5, "Importance_Change")
    most_important_lost_feature = most_reduced_features.iloc[0]["Feature"]

    print(f"\n✅ The most affected feature was: **{most_important_lost_feature}**, losing {most_reduced_features.iloc[0]['Importance_Change']:.4f} importance.")

    if most_reduced_features["Importance_Change"].min() < -0.05:
        print("⚠️ Anonymization significantly reduced the importance of key predictive features.")
    else:
        print("✅ Anonymization had minimal impact on feature importance.")

    print("\n===== TEST COMPLETE: Feature Importance Shift Analysis Finished =====\n")



