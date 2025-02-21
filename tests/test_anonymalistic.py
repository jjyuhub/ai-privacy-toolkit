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

def test_anonymize_pandas_adult():
    """
    This test ensures that k-anonymization is properly applied to the Adult dataset.
    It performs the following steps in a verbose and detailed manner:
    1. Loads the dataset
    2. Defines features and quasi-identifiers (QI) that will be anonymized
    3. Preprocesses the data by handling missing values and encoding categorical features
    4. Trains a Decision Tree classifier to generate predictions
    5. Applies k-anonymization to generalize the dataset
    6. Verifies that quasi-identifiers (QI) have reduced uniqueness
    7. Ensures that k-anonymization enforces groups of at least 'k' individuals
    8. Confirms that non-QI features remain unchanged
    """
    
    # Step 1: Load the Adult dataset (Pandas format with labeled data)
    (x_train, y_train), _ = get_adult_dataset_pd()
    print(f"Dataset loaded with {x_train.shape[0]} rows and {x_train.shape[1]} columns.")
    
    # Step 2: Define k-anonymity level (ensuring each unique combination appears at least 'k' times)
    k = 100  # Every unique quasi-identifier group should contain at least 100 samples
    print(f"Applying k-anonymization with k={k}")
    
    # Step 3: Define dataset features and quasi-identifiers
    features = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    QI = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    
    print(f"Total features used: {len(features)}")
    print(f"Quasi-identifiers selected: {QI}")
    
    # Step 4: Preprocess numerical features by handling missing values
    numeric_features = [f for f in features if f not in categorical_features]  # Extract numerical feature names
    print(f"Numerical features identified: {numeric_features}")
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))])
    
    # Step 5: Convert categorical features into numerical format using one-hot encoding
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)  # Ignore unknown categories
    
    # Step 6: Apply transformations to categorical and numeric features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),  # Apply numeric preprocessing
            ("cat", categorical_transformer, categorical_features),  # Apply one-hot encoding
        ]
    )
    
    # Step 7: Transform input features
    encoded = preprocessor.fit_transform(x_train)  # Convert dataset to numerical format
    print(f"Feature transformation complete. Transformed dataset has {encoded.shape[0]} rows and {encoded.shape[1]} columns.")
    
    # Step 8: Train a Decision Tree classifier
    model = DecisionTreeClassifier()
    model.fit(encoded, y_train)  # Fit model to preprocessed data
    pred = model.predict(encoded)  # Generate predictions
    print("Decision Tree model trained and predictions generated.")
    
    # Step 9: Apply k-anonymization
    anonymizer = Anonymize(k, QI, categorical_features=categorical_features)
    anon = anonymizer.anonymize(ArrayDataset(x_train, pred, features))
    print("Anonymization process complete.")
    
    # Step 10: Ensure that unique QI combinations are reduced
    unique_qi_before = x_train.loc[:, QI].drop_duplicates().shape[0]
    unique_qi_after = anon.loc[:, QI].drop_duplicates().shape[0]
    print(f"Unique QI combinations before anonymization: {unique_qi_before}")
    print(f"Unique QI combinations after anonymization: {unique_qi_after}")
    assert unique_qi_after < unique_qi_before, "Error: Anonymization did not reduce QI uniqueness!"
    
    # Step 11: Verify that each unique QI combination has at least 'k' samples
    min_group_size = anon.loc[:, QI].value_counts().min()
    print(f"Smallest group size after anonymization: {min_group_size}")
    assert min_group_size >= k, f"Error: Some groups have fewer than {k} samples!"
    
    # Step 12: Ensure that non-QI features remain unchanged
    unchanged_features = np.array_equal(anon.drop(QI, axis=1), x_train.drop(QI, axis=1))
    print("Verifying that non-QI features remain unchanged...")
    assert unchanged_features, "Error: Non-QI features were modified during anonymization!"
    print("Test passed: k-anonymization successfully applied with non-QI features preserved.")
