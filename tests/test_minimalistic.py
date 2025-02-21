import pytest
import numpy as np
import pandas as pd
import scipy

from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from torch import nn, optim, sigmoid, where
from torch.nn import functional
from scipy.special import expit

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

from apt.utils.datasets.datasets import PytorchData
from apt.utils.models.pytorch_model import PyTorchClassifier
from apt.minimization import GeneralizeToRepresentative
from apt.utils.dataset_utils import get_iris_dataset_np, get_adult_dataset_pd, get_german_credit_dataset_pd
from apt.utils.datasets import ArrayDataset
from apt.utils.models import SklearnClassifier, SklearnRegressor, KerasClassifier, \
    CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES, CLASSIFIER_SINGLE_OUTPUT_CATEGORICAL, \
    CLASSIFIER_SINGLE_OUTPUT_CLASS_LOGITS, CLASSIFIER_MULTI_OUTPUT_BINARY_LOGITS

tf.compat.v1.disable_eager_execution()

ACCURACY_DIFF = 0.05  # Maximum allowed accuracy difference after anonymization

def test_minimizer_fit(data_two_features):
    """
    Test the GeneralizeToRepresentative minimization process on a dataset with two features.
    This function:
    - Loads dataset
    - Trains a DecisionTreeClassifier
    - Applies generalization minimization
    - Compares the expected vs actual results
    - Prints before and after transformations
    """
    x, y, features, _ = data_two_features  # Extract dataset information

    # Initialize the decision tree classifier
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2, min_samples_leaf=1)
    
    # Wrap the model with SklearnClassifier
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    
    # Fit the model to the dataset
    print("Training Decision Tree Classifier...")
    model.fit(ArrayDataset(x, y))
    
    # Convert dataset into ArrayDataset format for predictions
    ad = ArrayDataset(x)
    predictions = model.predict(ad)
    
    # Ensure single-label output if applicable
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    
    target_accuracy = 0.5  # Set a target accuracy threshold
    
    # Initialize the generalization minimizer
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy)
    
    # Convert the dataset into training format
    train_dataset = ArrayDataset(x, predictions, features_names=features)
    
    # Print first 20 rows before transformation
    print("First 20 samples BEFORE transformation:")
    print(pd.DataFrame(x[:20], columns=features))
    
    # Fit the generalization model
    print("Fitting GeneralizeToRepresentative minimization...")
    gen.fit(dataset=train_dataset)
    
    # Transform the dataset
    transformed = gen.transform(dataset=ad)
    
    # Print first 20 rows after transformation
    print("First 20 samples AFTER transformation:")
    print(pd.DataFrame(transformed[:20], columns=features))
    
    # Retrieve generalizations from the transformation process
    gener = gen.generalizations
    expected_generalizations = {'ranges': {}, 'categories': {}, 'untouched': ['height', 'age']}
    
    # Compare expected generalizations with actual generalizations
    compare_generalizations(gener, expected_generalizations)
    
    # Check if features are transformed correctly
    check_features(features, expected_generalizations, transformed, x)
    
    # Ensure that transformation did not change original data values
    assert np.equal(x, transformed).all(), "Transformed data must match original data"
    
    # Calculate normalized certainty penalty (NCP)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)
    
    # Compute and validate relative accuracy after transformation
    rel_accuracy = model.score(ArrayDataset(transformed, predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)
    
    print("Test completed successfully!")
