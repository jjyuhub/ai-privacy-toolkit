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

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, f_oneway
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from apt.anonymization import Anonymize
from apt.utils.dataset_utils import get_adult_dataset_pd
from apt.utils.datasets import ArrayDataset

def test_feature_importance_shift_multiple_k_with_stats():
    """
    Compares feature importance shifts for k-anonymization with k=5, k=10, and k=100.
    Uses T-Tests and ANOVA to assess statistical significance of feature importance shifts.
    """

    print("\n===== STARTING TEST: Feature Importance Shift Across k-Values with Statistical Tests =====\n")

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
    importance_results = {}

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

        # Store results
        importance_results[k] = feature_importance_after

    # Step 6: Statistical Significance Tests
    print("\n===== Statistical Significance Analysis =====\n")

    # Perform T-Test (paired) between original and each k-anonymized dataset
    for k in k_values:
        t_stat, p_value = ttest_rel(feature_importance_before, importance_results[k])
        print(f"T-Test for k={k}: t-statistic={t_stat:.4f}, p-value={p_value:.4f}")

        if p_value < 0.05:
            print(f"✅ Statistically Significant Change Detected for k={k} (p < 0.05)")
        else:
            print(f"⚠️ No Significant Change Detected for k={k} (p >= 0.05)")

    # Perform ANOVA across different k-values
    f_stat, anova_p_value = f_oneway(
        feature_importance_before, importance_results[5], importance_results[10], importance_results[100]
    )

    print(f"\nANOVA Test Across k-values: F-statistic={f_stat:.4f}, p-value={anova_p_value:.4f}")

    if anova_p_value < 0.05:
        print("✅ Statistically Significant Differences in Feature Importance Across k-values (p < 0.05)")
    else:
        print("⚠️ No Significant Differences Across k-values (p >= 0.05)")

    print("\n===== TEST COMPLETE: Feature Importance Shift with Statistical Analysis =====\n")









