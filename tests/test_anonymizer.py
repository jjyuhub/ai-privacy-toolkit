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

def test_anonymize_ndarray_iris():
    """
    Verbose test for anonymizing the Iris dataset.
    Ensures each step prints information before assertions are checked.
    """

    # ---------------------- STEP 1: LOAD DATASET ----------------------
    print("\n[STEP 1] Loading Iris dataset...")  # Explicit step logging
    (x_train, y_train), _ = get_iris_dataset_np()

    print(f"  Dataset Shape: {x_train.shape}")  # (150, 4)
    print(f"  First 5 Samples:\n{x_train[:5]}")  # Print example data
    print(f"  Unique Labels: {np.unique(y_train)}\n")  # [0, 1, 2]

    # ---------------------- STEP 2: TRAIN DECISION TREE ----------------------
    print("[STEP 2] Training Decision Tree Model...")
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    # Predict using the trained model
    pred = model.predict(x_train)
    print(f"  First 5 Predictions: {pred[:5]}\n")

    # ---------------------- STEP 3: APPLY ANONYMIZATION ----------------------
    print("[STEP 3] Applying Anonymization...")
    k = 10  # Define k-anonymity level
    QI = [0, 2]  # Define Quasi-Identifiers (QI): sepal length and petal length

    print(f"  Unique QI Values Before Anonymization: {len(np.unique(x_train[:, QI], axis=0))}")

    # Initialize and apply anonymization
    anonymizer = Anonymize(k, QI, train_only_QI=True)
    anon = anonymizer.anonymize(ArrayDataset(x_train, pred))

    print(f"  First 5 Rows After Anonymization:\n{anon[:5]}\n")

    # ---------------------- STEP 4: VALIDATE ANONYMIZATION ----------------------
    print("[STEP 4] Validating Anonymization...")

    # Check that QI uniqueness is reduced
    unique_qi_before = len(np.unique(x_train[:, QI], axis=0))
    unique_qi_after = len(np.unique(anon[:, QI], axis=0))

    print(f"  Unique QI Values Before: {unique_qi_before}")
    print(f"  Unique QI Values After: {unique_qi_after}")
    assert unique_qi_after < unique_qi_before, "Anonymization did not reduce unique QI values!"

    # Check that each group has at least 'k' records
    _, counts_elements = np.unique(anon[:, QI], return_counts=True)

    print(f"  Minimum group size after anonymization: {np.min(counts_elements)}")
    print(f"  All group sizes: {counts_elements}")
    assert np.min(counts_elements) >= k, "Some anonymized groups contain fewer than 'k' samples!"

    # Ensure non-QI features remain unchanged
    non_qi_unchanged = (np.delete(anon, QI, axis=1) == np.delete(x_train, QI, axis=1)).all()

    print(f"  Non-QI Feature Integrity Check: {non_qi_unchanged}")
    assert non_qi_unchanged, "Non-QI features were modified!"

    print("\n✅ TEST PASSED: Anonymization applied correctly!\n")


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



def test_anonymize_pandas_nursery():
    """
    This test applies k-anonymization to the Nursery dataset to ensure that quasi-identifiers (QI) 
    are generalized while preserving the dataset’s structure. The test prints each step, displaying 
    intermediate outputs to help understand the anonymization process.
    """

    # Step 1: Load the Nursery dataset (Pandas format)
    print("\n[Step 1] Loading the Nursery dataset...")
    (x_train, y_train), _ = get_nursery_dataset_pd()
    print(f" - Number of records in dataset: {x_train.shape[0]}")
    print(f" - Number of features in dataset: {x_train.shape[1]}")

    # Step 2: Convert all feature values to strings (ensuring categorical consistency)
    print("\n[Step 2] Converting all feature values to strings to ensure categorical consistency...")
    x_train = x_train.astype(str)

    # Step 3: Define k-anonymity level (ensuring that each QI group has at least 'k' records)
    k = 100
    print(f"\n[Step 3] Setting k-anonymity level: k = {k}")

    # Step 4: Define all available features in the dataset
    features = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"]
    print(f" - Features in dataset: {features}")

    # Step 5: Define quasi-identifiers (QI) that will be anonymized
    QI = ["finance", "social", "health"]
    print(f" - Quasi-identifiers selected for anonymization: {QI}")

    # Step 6: Define categorical features in the dataset
    categorical_features = ["parents", "has_nurs", "form", "housing", "finance", "social", "health", "children"]
    print(f" - Categorical features: {categorical_features}")

    # Step 7: Identify numerical features (those not listed as categorical)
    numeric_features = [f for f in features if f not in categorical_features]
    print(f" - Numerical features: {numeric_features}")

    # Step 8: Preprocessing step for numerical features: Fill missing values with a constant value (0)
    print("\n[Step 8] Creating preprocessing pipeline for numerical features (imputing missing values)...")
    numeric_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))]
    )

    # Step 9: Preprocessing step for categorical features: Convert them to numerical format using one-hot encoding
    print("[Step 9] Creating preprocessing pipeline for categorical features (one-hot encoding)...")
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    # Step 10: Apply transformations: Numerical features are imputed, categorical features are one-hot encoded
    print("[Step 10] Combining preprocessing steps into a column transformer...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),  # Apply numeric processing
            ("cat", categorical_transformer, categorical_features),  # Apply categorical processing
        ]
    )

    # Step 11: Transform the dataset (convert categorical features to numerical format)
    print("[Step 11] Applying transformations to dataset...")
    encoded = preprocessor.fit_transform(x_train)
    print(f" - Transformed dataset shape: {encoded.shape}")

    # Step 12: Train a Decision Tree classifier on the transformed dataset
    print("\n[Step 12] Training Decision Tree classifier on transformed data...")
    model = DecisionTreeClassifier()
    model.fit(encoded, y_train)

    # Step 13: Generate predictions from the trained model
    print("[Step 13] Generating predictions using trained model...")
    pred = model.predict(encoded)
    print(f" - Example predictions: {pred[:10]}")

    # Step 14: Initialize the anonymizer with k-anonymity, targeting the selected quasi-identifiers
    print("\n[Step 14] Initializing anonymizer with k-anonymity constraints...")
    anonymizer = Anonymize(k, QI, categorical_features=categorical_features, train_only_QI=True)

    # Step 15: Apply anonymization: Generalizing quasi-identifiers while preserving model output
    print("[Step 15] Applying anonymization to dataset...")
    anon = anonymizer.anonymize(ArrayDataset(x_train, pred))
    print(f" - Anonymized dataset shape: {anon.shape}")

    # Step 16: Verify that the number of unique QI groups is reduced after anonymization
    unique_qi_before = x_train.loc[:, QI].drop_duplicates().shape[0]
    unique_qi_after = anon.loc[:, QI].drop_duplicates().shape[0]
    print("\n[Step 16] Checking if the number of unique QI groups has been reduced...")
    print(f" - Unique QI groups before anonymization: {unique_qi_before}")
    print(f" - Unique QI groups after anonymization: {unique_qi_after}")
    assert unique_qi_after < unique_qi_before, "Anonymization did not reduce unique QI groups."

    # Step 17: Ensure that each quasi-identifier group has at least 'k' records (enforcing k-anonymity)
    print("\n[Step 17] Verifying that each quasi-identifier group has at least 'k' records...")
    min_group_size = anon.loc[:, QI].value_counts().min()
    print(f" - Smallest QI group size after anonymization: {min_group_size}")
    assert min_group_size >= k, f"Some groups contain fewer than {k} records!"

    # Step 18: Validate that non-QI attributes remain unchanged after anonymization
    print("\n[Step 18] Checking if non-QI attributes remain unchanged...")
    unchanged_attributes = np.array_equal(anon.drop(QI, axis=1), x_train.drop(QI, axis=1))
    print(f" - Non-QI attributes remain unchanged: {unchanged_attributes}")
    np.testing.assert_array_equal(anon.drop(QI, axis=1), x_train.drop(QI, axis=1))

    print("\n✅ Test Passed: Anonymization works as expected!")

def test_regression():
    """
    Test function to train a Decision Tree Regressor on the Diabetes dataset,
    apply k-anonymization, and evaluate model performance before and after anonymization.
    
    This function prints detailed logs at every step and displays sample data before and after processing.
    """
    
    # Step 1: Load the diabetes dataset
    print("Loading the diabetes dataset...")
    dataset = load_diabetes()
    
    # Step 2: Split the dataset into training (50%) and testing (50%) sets
    print("Splitting dataset into training (50%) and testing (50%)...")
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.5, random_state=14)
    
    print("Training set size:", x_train.shape)
    print("Testing set size:", x_test.shape)
    
    # Step 3: Train a Decision Tree Regressor
    print("Training Decision Tree Regressor on the training data...")
    model = DecisionTreeRegressor(random_state=10, min_samples_split=2)
    model.fit(x_train, y_train)
    
    # Step 4: Generate predictions on the training set
    print("Generating predictions on training data...")
    pred = model.predict(x_train)
    print("First 5 predictions:", pred[:5])
    
    # Step 5: Define k-anonymity level and Quasi-Identifiers (QI)
    k = 10  # Minimum group size for anonymization
    QI = [0, 2, 5, 8]  # Selecting features (indices) to be anonymized
    
    print("Applying k-anonymization with k =", k)
    print("Quasi-Identifiers (QI) selected:", QI)
    
    # Step 6: Apply anonymization to the dataset
    anonymizer = Anonymize(k, QI, is_regression=True, train_only_QI=True)
    anon = anonymizer.anonymize(ArrayDataset(x_train, pred))
    
    print("Anonymization complete. Checking results...")
    
    # Step 7: Evaluate model performance before anonymization
    base_score = model.score(x_test, y_test)
    print("Base model accuracy (R2 score) before anonymization:", base_score)
    
    # Step 8: Retrain the model using anonymized data
    model.fit(anon, y_train)
    anon_score = model.score(x_test, y_test)
    print("Base model accuracy (R2 score) after anonymization:", anon_score)
    
    # Step 9: Verify anonymization effectiveness
    
    # Ensure that the number of unique QI values is reduced
    unique_before = len(np.unique(x_train[:, QI], axis=0))
    unique_after = len(np.unique(anon[:, QI], axis=0))
    print("Unique QI combinations before anonymization:", unique_before)
    print("Unique QI combinations after anonymization:", unique_after)
    assert unique_after < unique_before, "Anonymization did not reduce unique QI values."
    
    # Ensure that each anonymized group contains at least k samples
    _, counts_elements = np.unique(anon[:, QI], return_counts=True)
    min_group_size = np.min(counts_elements)
    print("Minimum group size after anonymization:", min_group_size)
    assert min_group_size >= k, "Anonymization failed: Some groups have fewer than k elements."
    
    # Ensure non-QI features remain unchanged
    unchanged_features = np.delete(anon, QI, axis=1) == np.delete(x_train, QI, axis=1)
    print("Checking if non-QI features remain unchanged...")
    assert unchanged_features.all(), "Non-QI features were modified during anonymization."
    
    # Step 10: Print first 5 samples before and after anonymization for visual comparison
    print("\nFirst 5 samples BEFORE anonymization:")
    print(pd.DataFrame(x_train[:5], columns=[f"Feature {i}" for i in range(x_train.shape[1])]))
    
    print("\nFirst 5 samples AFTER anonymization:")
    print(pd.DataFrame(anon[:5], columns=[f"Feature {i}" for i in range(x_train.shape[1])]))
    
    print("Test completed successfully.")


def test_anonymize_ndarray_one_hot():
    """
    Test function to apply k-anonymization on a NumPy array with one-hot encoded quasi-identifiers (QI).
    Ensures that the number of unique QI values is reduced and maintains the integrity of non-QI features.
    This function provides **detailed** logging for each step and prints intermediate outputs.
    """

    print("\n===== STARTING TEST: Anonymizing One-Hot Encoded NumPy Data =====\n")

    # Step 1: Define the training dataset (features)
    # Each row represents a data entry with four features:
    # - First column: Age (numeric)
    # - Second & Third columns: One-hot encoded categorical feature (e.g., gender: male=0, female=1)
    # - Fourth column: Another numeric feature (e.g., height)
    x_train = np.array([
        [23, 0, 1, 165],  # Male, 23 years old, 165 cm
        [45, 0, 1, 158],  # Male, 45 years old, 158 cm
        [56, 1, 0, 123],  # Female, 56 years old, 123 cm
        [67, 0, 1, 154],  # Male, 67 years old, 154 cm
        [45, 1, 0, 149],  # Female, 45 years old, 149 cm
        [42, 1, 0, 166],  # Female, 42 years old, 166 cm
        [73, 0, 1, 172],  # Male, 73 years old, 172 cm
        [94, 0, 1, 168],  # Male, 94 years old, 168 cm
        [69, 0, 1, 175],  # Male, 69 years old, 175 cm
        [24, 1, 0, 181],  # Female, 24 years old, 181 cm
        [18, 1, 0, 190],  # Female, 18 years old, 190 cm
    ])

    # Step 2: Define the target labels (binary classification)
    # These labels indicate whether a sample belongs to class 1 or class 0.
    y_train = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

    # Print the first 20 original data entries before processing
    print("\nOriginal Data (First 20 Samples):\n", x_train)

    # Step 3: Train a Decision Tree model on the dataset
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)  # Train the model using x_train as input and y_train as labels

    # Step 4: Generate predictions using the trained model
    pred = model.predict(x_train)

    # Print the first 20 predicted labels
    print("\nPredicted Labels (First 20 Samples):\n", pred)

    # Step 5: Define k-anonymity threshold (minimum number of samples per anonymized group)
    k = 10  # Each group must have at least k samples

    # Step 6: Define quasi-identifiers (QI) - features that should be anonymized
    QI = [0, 1, 2]  # Choosing Age, Gender (one-hot), and another one-hot feature as quasi-identifiers

    # Step 7: Define QI slices (groups of features that should be treated together for anonymization)
    QI_slices = [[1, 2]]  # The second and third columns (one-hot encoded) should be anonymized together

    # Step 8: Initialize the Anonymizer with k-anonymity enforcement
    anonymizer = Anonymize(k, QI, train_only_QI=True, quasi_identifer_slices=QI_slices)

    print("\nApplying Anonymization Process...\n")

    # Step 9: Apply anonymization to the dataset
    anon = anonymizer.anonymize(ArrayDataset(x_train, pred))

    # Step 10: Print the first 20 anonymized samples
    print("\nAnonymized Data (First 20 Samples):\n", anon[:20])

    # Step 11: Check if the number of unique QI combinations has been reduced
    unique_QI_before = len(np.unique(x_train[:, QI], axis=0))
    unique_QI_after = len(np.unique(anon[:, QI], axis=0))
    print("\nUnique QI Combinations Before Anonymization:", unique_QI_before)
    print("Unique QI Combinations After Anonymization:", unique_QI_after)

    # Assertion 1: Ensure that the number of unique QI combinations is reduced
    assert unique_QI_after < unique_QI_before, "Anonymization failed to reduce QI uniqueness"

    # Step 12: Count occurrences of each unique QI combination in the anonymized dataset
    _, counts_elements = np.unique(anon[:, QI], return_counts=True)

    # Print the distribution of records per unique QI combination
    print("\nAnonymized Data - Frequency of Unique QI Combinations:\n", counts_elements)

    # Step 13: Ensure that each group has at least 'k' records (k-anonymity constraint)
    min_group_size = np.min(counts_elements)
    print("\nSmallest Anonymized Group Size:", min_group_size)
    assert min_group_size >= k, "Anonymization failed: Some groups have fewer than k records"

    # Step 14: Ensure that non-QI features remain unchanged
    unchanged_features_check = np.delete(anon, QI, axis=1) == np.delete(x_train, QI, axis=1)
    print("\nNon-QI Features Unchanged:", np.all(unchanged_features_check))
    assert np.all(unchanged_features_check), "Anonymization incorrectly modified non-QI features"

    # Step 15: Extract the anonymized slice of QI values that were grouped together
    anonymized_slice = anon[:, QI_slices[0]]

    # Step 16: Ensure that one-hot encoded QI slices still represent valid categories
    sum_of_each_row = np.sum(anonymized_slice, axis=1)
    max_value_each_row = np.max(anonymized_slice, axis=1)
    min_value_each_row = np.min(anonymized_slice, axis=1)

    # Print one-hot encoding validation checks
    print("\nOne-Hot Encoding Check:")
    print(" - Sum of each row (should be 1):", sum_of_each_row)
    print(" - Max value per row (should be 1):", max_value_each_row)
    print(" - Min value per row (should be 0):", min_value_each_row)

    # Assertion 4: Each row should have exactly **one** active category (sum should be 1)
    assert (sum_of_each_row == 1).all(), "One-hot encoding constraint violated: Row sum is not 1"

    # Assertion 5: Each row should have a max value of 1 (valid one-hot encoding)
    assert (max_value_each_row == 1).all(), "One-hot encoding constraint violated: Max value per row is not 1"

    # Assertion 6: Each row should have a min value of 0 (valid one-hot encoding)
    assert (min_value_each_row == 0).all(), "One-hot encoding constraint violated: Min value per row is not 0"

    print("\n===== TEST PASSED: Anonymization Successful and Data Integrity Maintained =====\n")




def test_anonymize_pandas_one_hot():
    """
    Test function to apply k-anonymization to a Pandas dataset 
    with one-hot encoded categorical features.
    """

    # Define the feature names for the dataset
    feature_names = ["age", "gender_M", "gender_F", "height"]

    # Create a NumPy array representing the dataset (11 rows, 4 columns)
    # Columns: Age (numeric), Gender_M (binary), Gender_F (binary), Height (numeric)
    x_train = np.array([[23, 0, 1, 165],  # Row 1: 23 years, Female, 165cm
                        [45, 0, 1, 158],  # Row 2: 45 years, Female, 158cm
                        [56, 1, 0, 123],  # Row 3: 56 years, Male, 123cm
                        [67, 0, 1, 154],  # Row 4: 67 years, Female, 154cm
                        [45, 1, 0, 149],  # Row 5: 45 years, Male, 149cm
                        [42, 1, 0, 166],  # Row 6: 42 years, Male, 166cm
                        [73, 0, 1, 172],  # Row 7: 73 years, Female, 172cm
                        [94, 0, 1, 168],  # Row 8: 94 years, Female, 168cm
                        [69, 0, 1, 175],  # Row 9: 69 years, Female, 175cm
                        [24, 1, 0, 181],  # Row 10: 24 years, Male, 181cm
                        [18, 1, 0, 190]]) # Row 11: 18 years, Male, 190cm

    # Create an array for the target labels (classification: binary 0/1)
    y_train = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

    # Convert the NumPy array into a Pandas DataFrame with column names
    x_train = pd.DataFrame(x_train, columns=feature_names)

    # Convert y_train into a Pandas Series (to match the DataFrame structure)
    y_train = pd.Series(y_train)

    # Initialize a Decision Tree classifier
    model = DecisionTreeClassifier()

    # Train the Decision Tree model on the dataset
    model.fit(x_train, y_train)

    # Generate predictions on the training set
    pred = model.predict(x_train)

    # Define k-anonymization parameter (each group must have at least k=10 similar records)
    k = 10

    # Define the quasi-identifiers (sensitive attributes that should be anonymized)
    QI = ["age", "gender_M", "gender_F"]

    # Define groups of quasi-identifiers that should be generalized together
    QI_slices = [["gender_M", "gender_F"]]  # Gender is one-hot encoded, so it should be treated as a single attribute

    # Initialize the Anonymize class to apply k-anonymization
    anonymizer = Anonymize(k, QI, train_only_QI=True, quasi_identifer_slices=QI_slices)

    # Apply anonymization to the dataset
    anon = anonymizer.anonymize(ArrayDataset(x_train, pred))

    # **Assertions to verify correctness of anonymization process**

    # Check that the number of unique values in the QI columns has been reduced after anonymization
    assert (anon.loc[:, QI].drop_duplicates().shape[0] < x_train.loc[:, QI].drop_duplicates().shape[0])

    # Check that each anonymized group contains at least 'k' samples (ensuring k-anonymity)
    assert (anon.loc[:, QI].value_counts().min() >= k)

    # Ensure that only the quasi-identifier columns have changed, while other features remain unchanged
    np.testing.assert_array_equal(anon.drop(QI, axis=1), x_train.drop(QI, axis=1))

    # Extract the anonymized slice for the one-hot encoded gender column
    anonymized_slice = anon.loc[:, QI_slices[0]]

    # Ensure that the sum of the one-hot encoded gender columns equals 1 for all rows
    assert ((np.sum(anonymized_slice, axis=1) == 1).all())  # Ensures a valid one-hot encoding

    # Ensure that the maximum value in the one-hot encoded columns is 1 (no invalid values)
    assert ((np.max(anonymized_slice, axis=1) == 1).all())

    # Ensure that the minimum value in the one-hot encoded columns is 0 (no invalid values)
    assert ((np.min(anonymized_slice, axis=1) == 0).all())



def test_anonymize_pytorch_multi_label_binary():
    """
    Test case for applying k-anonymization on a PyTorch multi-label binary classifier.
    Ensures the anonymized dataset reduces unique quasi-identifier values while preserving data consistency.
    """

    # Define a PyTorch model for multi-label binary classification
    class multi_label_binary_model(nn.Module):
        def __init__(self, num_labels, num_features):
            """
            Initializes a simple feedforward neural network for multi-label binary classification.
            Args:
                num_labels (int): The number of output labels (classes).
                num_features (int): The number of input features.
            """
            super(multi_label_binary_model, self).__init__()

            # First fully connected layer with 256 neurons and Tanh activation
            self.fc1 = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.Tanh(),
            )

            # Final classifier layer mapping from 256 neurons to output labels
            self.classifier1 = nn.Linear(256, num_labels)

        def forward(self, x):
            """
            Forward pass of the model.
            Args:
                x (tensor): Input data.
            Returns:
                Tensor: Model predictions (logits).
            """
            return self.classifier1(self.fc1(x))  # Missing sigmoid activation for binary classification

    # Define the Focal Loss function, useful for handling imbalanced datasets
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2, alpha=0.5):
            """
            Initializes Focal Loss to focus on hard-to-classify samples.
            Args:
                gamma (float): Controls how much focus is given to hard-to-classify examples.
                alpha (float): Balancing factor for positive and negative class weights.
            """
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.alpha = alpha

        def forward(self, input, target):
            """
            Computes Focal Loss.
            Args:
                input (tensor): Model predictions (logits).
                target (tensor): True labels.
            Returns:
                Tensor: Computed Focal Loss.
            """
            # Compute standard Binary Cross-Entropy (BCE) loss
            bce_loss = functional.binary_cross_entropy_with_logits(input, target, reduction='none')

            # Convert logits into probabilities using the sigmoid function
            p = sigmoid(input)

            # Adjust probabilities based on target labels
            p = where(target >= 0.5, p, 1 - p)

            # Apply Focal Loss adjustments: modulating factor reduces importance of easy examples
            modulating_factor = (1 - p) ** self.gamma
            alpha = self.alpha * target + (1 - self.alpha) * (1 - target)

            # Compute final Focal Loss value
            focal_loss = alpha * modulating_factor * bce_loss
            return focal_loss.mean()

    # Load the Iris dataset (NumPy format)
    (x_train, y_train), _ = get_iris_dataset_np()

    # Convert dataset to multi-label binary format by stacking the labels
    y_train = np.column_stack((y_train, y_train, y_train))

    # Convert all labels greater than 1 into 1 (binary classification: 0 or 1)
    y_train[y_train > 1] = 1

    # Initialize the multi-label binary classification model
    model = multi_label_binary_model(3, 4)

    # Define Focal Loss as the loss function
    criterion = FocalLoss()

    # Use RMSprop optimizer for updating model weights
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)

    # Wrap the model using the PyTorchClassifier utility for ART compatibility
    art_model = PyTorchClassifier(
        model=model,
        output_type=CLASSIFIER_MULTI_OUTPUT_BINARY_LOGITS,  # Specifies the type of classifier
        loss=criterion,  # Assigns Focal Loss function
        optimizer=optimizer,  # Assigns RMSprop optimizer
        input_shape=(24,),  # Defines the input feature shape
        nb_classes=3  # Number of output labels
    )

    # Convert dataset to PyTorch format and train the model for 10 epochs
    art_model.fit(PytorchData(x_train.astype(np.float32), y_train.astype(np.float32)), save_entire_model=False, nb_epochs=10)

    # Make predictions on the training dataset
    pred = art_model.predict(PytorchData(x_train.astype(np.float32), y_train.astype(np.float32)))

    # Apply the sigmoid function (expit) to convert logits to probabilities
    pred = expit(pred)

    # Convert probabilities into binary values (threshold at 0.5)
    pred[pred < 0.5] = 0  # If probability is below 0.5, set to 0
    pred[pred >= 0.5] = 1  # If probability is 0.5 or greater, set to 1

    # Define k-anonymization parameters
    k = 10  # Minimum number of records per anonymized group
    QI = [0, 2]  # Quasi-identifiers (features that need anonymization)

    # Initialize the anonymizer with specified k-anonymity parameters
    anonymizer = Anonymize(k, QI, train_only_QI=True)

    # Apply anonymization to the dataset
    anon = anonymizer.anonymize(ArrayDataset(x_train, pred))

    # Ensure that the number of unique quasi-identifier values decreases after anonymization
    assert (len(np.unique(anon[:, QI], axis=0)) < len(np.unique(x_train[:, QI], axis=0)))

    # Count the occurrences of each unique quasi-identifier value after anonymization
    _, counts_elements = np.unique(anon[:, QI], return_counts=True)

    # Verify that every anonymized group contains at least 'k' records
    assert (np.min(counts_elements) >= k)

    # Ensure that non-quasi-identifier features remain unchanged
    assert ((np.delete(anon, QI, axis=1) == np.delete(x_train, QI, axis=1)).all())



def test_errors():
    """
    This function tests various error cases to ensure that the Anonymize class
    and its methods raise the expected ValueErrors when incorrect parameters are passed.
    """

    # Test case 1: k-anonymity value of 1 is invalid (must be >= 2) with quasi-identifiers [0, 2]
    with pytest.raises(ValueError):  
        Anonymize(1, [0, 2])

    # Test case 2: Quasi-identifier list is empty, which is not allowed
    with pytest.raises(ValueError):  
        Anonymize(2, [])

    # Test case 3: Quasi-identifier list is None, which should raise an error
    with pytest.raises(ValueError):  
        Anonymize(2, None)

    # Create an anonymizer instance with valid parameters (k=10 and quasi-identifiers [0, 2])
    anonymizer = Anonymize(10, [0, 2])

    # Load the Iris dataset (NumPy format)
    (x_train, y_train), (x_test, y_test) = get_iris_dataset_np()

    # Test case 4: Passing mismatched labels (y_test instead of y_train) should raise a ValueError
    with pytest.raises(ValueError):  
        anonymizer.anonymize(dataset=ArrayDataset(x_train, y_test))

    # Load the Adult dataset (Pandas format)
    (x_train, y_train), _ = get_adult_dataset_pd()

    # Test case 5: Again, passing mismatched labels should raise a ValueError
    with pytest.raises(ValueError):  
        anonymizer.anonymize(dataset=ArrayDataset(x_train, y_test))

