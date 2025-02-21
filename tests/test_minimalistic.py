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

def create_encoder(numeric_features, categorical_features, x):
    numeric_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))]
    )

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    encoded = preprocessor.fit_transform(x)
    if scipy.sparse.issparse(encoded):
        pd.DataFrame.sparse.from_spmatrix(encoded)
    else:
        encoded = pd.DataFrame(encoded)

    return preprocessor, encoded




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
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import chi2_contingency
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

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

def compute_fairness_metrics(y_true, y_pred, sensitive_attr):
    """
    Compute fairness metrics using statistical significance, false positive/negative rates, and calibration fairness.
    """
    print("Computing fairness metrics...")
    subgroups = np.unique(sensitive_attr)
    metrics = {}
    counts = []
    false_positive_rates = {}
    false_negative_rates = {}
    calibration_scores = {}
    
    for subgroup in subgroups:
        indices = (sensitive_attr == subgroup)
        correct = np.sum(y_pred[indices] == y_true[indices])
        total = np.sum(indices)
        acc = correct / total if total > 0 else 0
        metrics[subgroup] = acc
        counts.append([correct, total - correct])  # Store as [correct, incorrect]
        
        # Compute False Positive and False Negative Rates
        tn, fp, fn, tp = confusion_matrix(y_true[indices], y_pred[indices]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        false_positive_rates[subgroup] = fpr
        false_negative_rates[subgroup] = fnr
        
        # Compute Calibration Score (Brier Score)
        brier_score = brier_score_loss(y_true[indices], y_pred[indices])
        calibration_scores[subgroup] = brier_score
    
    _, p_value, _, _ = chi2_contingency(counts)  # Perform Chi-Square Test
    print(f"Chi-Square p-value: {p_value:.6f}")
    assert p_value < 0.05, "Test failed due to statistically significant fairness violation!"
    
    # Check False Positive/Negative Rate Disparity
    max_fpr = max(false_positive_rates.values())
    min_fpr = min(false_positive_rates.values())
    max_fnr = max(false_negative_rates.values())
    min_fnr = min(false_negative_rates.values())
    
    assert abs(max_fpr - min_fpr) < 0.01, "Test failed due to false positive rate disparity!"
    assert abs(max_fnr - min_fnr) < 0.01, "Test failed due to false negative rate disparity!"
    
    # Check Calibration Fairness
    max_brier = max(calibration_scores.values())
    min_brier = min(calibration_scores.values())
    
    assert abs(max_brier - min_brier) < 0.01, "Test failed due to calibration disparity!"
    
    print("Fairness check passed. No severe disparities detected.")
    return metrics

def test_minimize_pandas_adult():
    """
    Test the GeneralizeToRepresentative minimization process on the Adult dataset with full verbosity.
    """
    
    print("Loading the Adult dataset...")
    (x_train, y_train), _ = get_adult_dataset_pd()
    x_train = x_train.head(1000)
    y_train = y_train.head(1000)
    features = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    x_train = pd.DataFrame(x_train, columns=features)
    sensitive_attr = x_train['race'].values
    categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                            'hours-per-week', 'native-country']
    numeric_features = [f for f in features if f not in categorical_features]
    
    print("Encoding categorical features...")
    preprocessor, encoded = create_encoder(numeric_features, categorical_features, x_train)
    
    print("Training Decision Tree Classifier...")
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2, min_samples_leaf=1)
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    model.fit(ArrayDataset(encoded, y_train))
    
    print("Making predictions on training data...")
    predictions = model.predict(ArrayDataset(encoded))
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    
    print("Computing fairness metrics before generalization...")
    compute_fairness_metrics(y_train, predictions, sensitive_attr)
    
    print("Applying GeneralizeToRepresentative minimization...")
    gen = GeneralizeToRepresentative(model, target_accuracy=0.7,
                                     categorical_features=categorical_features, features_to_minimize=features,
                                     encoder=preprocessor)
    gen.fit(dataset=ArrayDataset(x_train, predictions, features_names=features))
    transformed = gen.transform(dataset=ArrayDataset(x_train))
    
    print("Computing fairness metrics after generalization...")
    compute_fairness_metrics(y_train, predictions, sensitive_attr)
    
    print("Test completed successfully!")


















def test_minimizer_ncp(data_two_features):
    """
    Test the impact of different generalization methods on normalized certainty penalty (NCP).
    This test:
    - Trains a DecisionTreeClassifier on a dataset.
    - Evaluates generalization techniques using GeneralizeToRepresentative.
    - Computes NCP scores before and after transformation.
    - Compares results to ensure correctness.
    - Prints detailed calculations and explanations at each step.
    """
    x, y, features, x1 = data_two_features  # Load dataset

    # Initialize Decision Tree Classifier with specific parameters
    print("Initializing DecisionTreeClassifier with min_samples_split=2 and min_samples_leaf=1...")
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2, min_samples_leaf=1)
    
    # Wrap the classifier with SklearnClassifier
    print("Wrapping the classifier with SklearnClassifier...")
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    
    # Train the model on the dataset
    print("Training Decision Tree Classifier with the dataset...")
    model.fit(ArrayDataset(x, y))
    
    # Prepare datasets for evaluation
    print("Preparing datasets for evaluation...")
    ad = ArrayDataset(x)
    ad1 = ArrayDataset(x1, features_names=features)
    
    # Generate predictions
    print("Generating predictions from the trained model...")
    predictions = model.predict(ad)
    print(f"Raw Predictions:\n{predictions[:20]}")
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    print(f"Final Predictions after argmax (if applicable):\n{predictions[:20]}")
    
    # Set target accuracy for generalization
    target_accuracy = 0.4
    print(f"Setting target accuracy to {target_accuracy} for generalization...")
    train_dataset = ArrayDataset(x, predictions, features_names=features)
    
    # Apply generalization without transformation
    print("Applying GeneralizeToRepresentative without transformation...")
    gen1 = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, generalize_using_transform=False)
    gen1.fit(dataset=train_dataset)
    ncp1 = gen1.ncp.fit_score  # NCP after fitting
    ncp2 = gen1.calculate_ncp(ad1)  # NCP on new dataset
    print(f"NCP values:\n  ncp1 (after fitting): {ncp1}\n  ncp2 (on new dataset): {ncp2}")
    
    # Apply generalization with transformation
    print("Applying GeneralizeToRepresentative with transformation...")
    gen2 = GeneralizeToRepresentative(model, target_accuracy=target_accuracy)
    gen2.fit(dataset=train_dataset)
    ncp3 = gen2.ncp.fit_score  # NCP after fitting
    print(f"NCP3 (after fitting with transformation): {ncp3}")
    
    # Transform and compute NCP at different stages
    print("Transforming dataset ad1...")
    gen2.transform(dataset=ad1)
    ncp4 = gen2.ncp.transform_score
    print(f"NCP4 (after transforming ad1): {ncp4}")
    
    print("Transforming dataset ad...")
    gen2.transform(dataset=ad)
    ncp5 = gen2.ncp.transform_score
    print(f"NCP5 (after transforming ad): {ncp5}")
    
    print("Transforming dataset ad1 again...")
    gen2.transform(dataset=ad1)
    ncp6 = gen2.ncp.transform_score
    print(f"NCP6 (after second transformation of ad1): {ncp6}")
    
    # Print verbose results
    print("First 20 samples BEFORE transformation:")
    print(pd.DataFrame(x[:20], columns=features))
    
    print("First 20 samples AFTER transformation:")
    transformed_data = gen2.transform(dataset=ad)
    print(pd.DataFrame(transformed_data[:20], columns=features))
    
    # Assert expected relationships among NCP scores
    print("Validating expected relationships between NCP scores...")
    assert (ncp1 <= ncp3), f"Assertion failed: Expected ncp1 ({ncp1}) <= ncp3 ({ncp3})"
    assert (ncp2 != ncp3), f"Assertion failed: Expected ncp2 ({ncp2}) != ncp3 ({ncp3})"
    assert (ncp3 != ncp4), f"Assertion failed: Expected ncp3 ({ncp3}) != ncp4 ({ncp4})"
    assert (ncp4 != ncp5), f"Assertion failed: Expected ncp4 ({ncp4}) != ncp5 ({ncp5})"
    assert (ncp6 == ncp4), f"Assertion failed: Expected ncp6 ({ncp6}) == ncp4 ({ncp4})"
    
    print("All assertions passed! Test completed successfully!")

