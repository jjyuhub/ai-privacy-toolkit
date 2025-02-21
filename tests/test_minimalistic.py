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


ACCURACY_DIFF = 0.05


@pytest.fixture
def diabetes_dataset():
    return load_diabetes()


@pytest.fixture
def cells():
    cells = [{"id": 1, "ranges": {"age": {"start": None, "end": 38}, "height": {"start": None, "end": 170}}, "label": 0,
              'categories': {}, "representative": {"age": 26, "height": 149}},
             {"id": 2, "ranges": {"age": {"start": 39, "end": None}, "height": {"start": None, "end": 170}}, "label": 1,
              'categories': {}, "representative": {"age": 58, "height": 163}},
             {"id": 3, "ranges": {"age": {"start": None, "end": 38}, "height": {"start": 171, "end": None}}, "label": 0,
              'categories': {}, "representative": {"age": 31, "height": 184}},
             {"id": 4, "ranges": {"age": {"start": 39, "end": None}, "height": {"start": 171, "end": None}}, "label": 1,
              'categories': {}, "representative": {"age": 45, "height": 176}}
             ]
    features = ['age', 'height']
    x = np.array([[23, 165],
                  [45, 158],
                  [18, 190]])
    y = [1, 1, 0]
    return cells, features, x, y


@pytest.fixture
def cells_categorical():
    cells = [{'id': 1, 'label': 0, 'ranges': {'age': {'start': None, 'end': None}},
              'categories': {'sex': ['f', 'm']}, 'hist': [2, 0],
              'representative': {'age': 45, 'height': 149, 'sex': 'f'},
              'untouched': ['height']},
             {'id': 3, 'label': 1, 'ranges': {'age': {'start': None, 'end': None}},
              'categories': {'sex': ['f', 'm']}, 'hist': [0, 3],
              'representative': {'age': 23, 'height': 165, 'sex': 'f'},
              'untouched': ['height']},
             {'id': 4, 'label': 0, 'ranges': {'age': {'start': None, 'end': None}},
              'categories': {'sex': ['f', 'm']}, 'hist': [1, 0],
              'representative': {'age': 18, 'height': 190, 'sex': 'm'},
              'untouched': ['height']}
             ]
    features = ['age', 'height', 'sex']
    x = [[23, 165, 'f'],
         [45, 158, 'f'],
         [56, 123, 'f'],
         [67, 154, 'm'],
         [45, 149, 'f'],
         [42, 166, 'm'],
         [73, 172, 'm'],
         [94, 168, 'f'],
         [69, 175, 'm'],
         [24, 181, 'm'],
         [18, 190, 'm']]
    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    return cells, features, x, y


@pytest.fixture
def data_two_features():
    x = np.array([[23, 165],
                  [45, 158],
                  [56, 123],
                  [67, 154],
                  [45, 149],
                  [42, 166],
                  [73, 172],
                  [94, 168],
                  [69, 175],
                  [24, 181],
                  [18, 190]])
    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    x1 = np.array([[33, 165],
                   [43, 150],
                   [71, 143],
                   [92, 194],
                   [13, 125],
                   [22, 169]])
    features = ['age', 'height']
    return x, y, features, x1


@pytest.fixture
def data_three_features():
    features = ['age', 'height', 'weight']
    x = np.array([[23, 165, 70],
                  [45, 158, 67],
                  [56, 123, 65],
                  [67, 154, 90],
                  [45, 149, 67],
                  [42, 166, 58],
                  [73, 172, 68],
                  [94, 168, 69],
                  [69, 175, 80],
                  [24, 181, 95],
                  [18, 190, 102]])
    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    return x, y, features


@pytest.fixture
def data_four_features():
    features = ['age', 'height', 'sex', 'ola']
    x = [[23, 165, 'f', 'aa'],
         [45, 158, 'f', 'aa'],
         [56, 123, 'f', 'bb'],
         [67, 154, 'm', 'aa'],
         [45, 149, 'f', 'bb'],
         [42, 166, 'm', 'bb'],
         [73, 172, 'm', 'bb'],
         [94, 168, 'f', 'aa'],
         [69, 175, 'm', 'aa'],
         [24, 181, 'm', 'bb'],
         [18, 190, 'm', 'bb']]
    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    x1 = [[33, 165, 'f', 'aa'],
          [43, 150, 'm', 'aa'],
          [71, 143, 'f', 'aa'],
          [92, 194, 'm', 'aa'],
          [13, 125, 'f', 'aa'],
          [22, 169, 'f', 'bb']]
    return x, y, features, x1


@pytest.fixture
def data_five_features():
    features = ['age', 'height', 'weight', 'sex', 'ola']
    x = [[23, 165, 65, 'f', 'aa'],
         [45, 158, 76, 'f', 'aa'],
         [56, 123, 78, 'f', 'bb'],
         [67, 154, 87, 'm', 'aa'],
         [45, 149, 45, 'f', 'bb'],
         [42, 166, 76, 'm', 'bb'],
         [73, 172, 85, 'm', 'bb'],
         [94, 168, 92, 'f', 'aa'],
         [69, 175, 95, 'm', 'aa'],
         [24, 181, 49, 'm', 'bb'],
         [18, 190, 69, 'm', 'bb']]
    y = pd.Series([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    return x, y, features


def compare_generalizations(gener, expected_generalizations):
    for key in expected_generalizations['ranges']:
        assert (set(expected_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expected_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expected_generalizations['categories'][key]])
                == set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expected_generalizations['untouched']) == set(gener['untouched']))
    if 'range_representatives' in expected_generalizations:
        for key in expected_generalizations['range_representatives']:
            assert (set(expected_generalizations['range_representatives'][key])
                    == set(gener['range_representatives'][key]))
    if 'category_representatives' in expected_generalizations:
        for key in expected_generalizations['category_representatives']:
            assert (set(expected_generalizations['category_representatives'][key])
                    == set(gener['category_representatives'][key]))


def check_features(features, expected_generalizations, transformed, x, pandas=False):
    modified_features = [f for f in features if
                         f in expected_generalizations['categories'].keys() or f in expected_generalizations[
                             'ranges'].keys()]

    if pandas:
        np.testing.assert_array_equal(transformed.drop(modified_features, axis=1), x.drop(modified_features, axis=1))
        if len(expected_generalizations['ranges'].keys()) > 0 or len(expected_generalizations['categories'].keys()) > 0:
            assert (((transformed[modified_features]).equals(x[modified_features])) is False)
    else:
        indexes = []
        for i in range(len(features)):
            if features[i] in modified_features:
                indexes.append(i)
        if len(indexes) != transformed.shape[1]:
            assert (np.array_equal(np.delete(transformed, indexes, axis=1), np.delete(x, indexes, axis=1)))
        if len(expected_generalizations['ranges'].keys()) > 0 or len(expected_generalizations['categories'].keys()) > 0:
            assert (not np.array_equal(transformed[:, indexes], x[:, indexes]))


def check_ncp(ncp, expected_generalizations):
    if len(expected_generalizations['ranges'].keys()) > 0 or len(expected_generalizations['categories'].keys()) > 0:
        assert (ncp > 0.0)


def test_minimizer_ncp(data_two_features):
    """
    Test the impact of different generalization methods on normalized certainty penalty (NCP).
    This test:
    - Trains a DecisionTreeClassifier on a dataset.
    - Evaluates generalization techniques using GeneralizeToRepresentative.
    - Computes NCP scores before and after transformation.
    - Compares results to ensure correctness.
    """
    x, y, features, x1 = data_two_features  # Load dataset

    # Initialize Decision Tree Classifier with specific parameters
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2, min_samples_leaf=1)
    
    # Wrap the classifier with SklearnClassifier
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    
    # Train the model on the dataset
    print("Training Decision Tree Classifier...")
    model.fit(ArrayDataset(x, y))
    
    # Prepare datasets for evaluation
    ad = ArrayDataset(x)
    ad1 = ArrayDataset(x1, features_names=features)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict(ad)
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    
    # Set target accuracy for generalization
    target_accuracy = 0.4
    train_dataset = ArrayDataset(x, predictions, features_names=features)
    
    # Apply generalization without transformation
    print("Applying GeneralizeToRepresentative without transformation...")
    gen1 = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, generalize_using_transform=False)
    gen1.fit(dataset=train_dataset)
    ncp1 = gen1.ncp.fit_score  # NCP after fitting
    ncp2 = gen1.calculate_ncp(ad1)  # NCP on new dataset
    
    # Apply generalization with transformation
    print("Applying GeneralizeToRepresentative with transformation...")
    gen2 = GeneralizeToRepresentative(model, target_accuracy=target_accuracy)
    gen2.fit(dataset=train_dataset)
    ncp3 = gen2.ncp.fit_score  # NCP after fitting
    
    # Transform and compute NCP at different stages
    gen2.transform(dataset=ad1)
    ncp4 = gen2.ncp.transform_score
    gen2.transform(dataset=ad)
    ncp5 = gen2.ncp.transform_score
    gen2.transform(dataset=ad1)
    ncp6 = gen2.ncp.transform_score
    
    # Print verbose results
    print("First 20 samples BEFORE transformation:")
    print(pd.DataFrame(x[:20], columns=features))
    print("First 20 samples AFTER transformation:")
    transformed_data = gen2.transform(dataset=ad)
    print(pd.DataFrame(transformed_data[:20], columns=features))
    
    # Assert expected relationships among NCP scores
    assert (ncp1 <= ncp3), "Expected ncp1 <= ncp3, but found otherwise."
    assert (ncp2 != ncp3), "Expected ncp2 != ncp3, but found otherwise."
    assert (ncp3 != ncp4), "Expected ncp3 != ncp4, but found otherwise."
    assert (ncp4 != ncp5), "Expected ncp4 != ncp5, but found otherwise."
    assert (ncp6 == ncp4), "Expected ncp6 == ncp4, but found otherwise."
    
    print("Test completed successfully!")

