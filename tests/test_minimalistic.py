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
from sklearn.metrics import accuracy_score

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
NUM_SPLITS = 10  # Number of random splits for fairness stability testing

def compute_fairness_metrics(y_true, y_pred, sensitive_attr, intersectional_attr):
    """
    Compute fairness metrics including accuracy per subgroup and disparate impact.
    Also performs intersectional fairness analysis correctly.
    """
    print("Computing fairness metrics...")
    subgroups = np.unique(sensitive_attr)
    intersectional_groups = np.unique(list(zip(sensitive_attr, intersectional_attr)))
    
    print(f"Identified {len(subgroups)} subgroups in sensitive attribute.")
    print(f"Identified {len(np.unique(intersectional_attr))} unique gender categories: {np.unique(intersectional_attr)}")
    metrics = {}
    
    for subgroup in subgroups:
        indices = (sensitive_attr == subgroup)
        num_samples = np.sum(indices)
        print(f"Total samples in subgroup {subgroup}: {num_samples}")
        
        if num_samples > 0:
            acc = accuracy_score(y_true[indices], y_pred[indices])
            print(f"Accuracy for {subgroup}: {acc:.4f}")
        else:
            acc = 0
            print(f"No samples available for subgroup {subgroup}, setting accuracy to 0.")
        
        metrics[subgroup] = acc
    
    min_acc = min(metrics.values())
    max_acc = max(metrics.values())
    disparate_impact = min_acc / max_acc if max_acc > 0 else 0
    
    print("Final Fairness Metrics:")
    print("Accuracy per subgroup:", metrics)
    print(f"Disparate Impact (min/max accuracy ratio): {disparate_impact:.4f}")
    
    if disparate_impact < 0.8:
        print("WARNING: Disparate impact is below the acceptable threshold of 0.8, indicating potential unfairness.")
    else:
        print("Fairness check passed. No severe disparities detected.")
    
    # Intersectional Fairness Analysis
    print("Performing intersectional fairness analysis...")
    intersectional_metrics = {}
    
    for group in intersectional_groups:
        indices = (sensitive_attr == group[0]) & (intersectional_attr == group[1])
        num_samples = np.sum(indices)
        print(f"Total samples in intersectional subgroup {group}: {num_samples}")
        
        if num_samples > 0:
            acc = accuracy_score(y_true[indices], y_pred[indices])
            print(f"Accuracy for {group}: {acc:.4f}")
        else:
            acc = 0
            print(f"No samples available for {group}, setting accuracy to 0.")
        
        intersectional_metrics[group] = acc
    
    print("Final Intersectional Fairness Metrics:")
    print(intersectional_metrics)
    
    return metrics, disparate_impact, intersectional_metrics

def test_fairness_stability():
    """
    Perform fairness stability testing over multiple random dataset splits with intersectional fairness analysis and test set evaluation.
    """
    print("Running fairness stability test with multiple dataset splits...")
    disparate_impact_scores = []
    
    for i in range(NUM_SPLITS):
        print(f"Iteration {i+1}/{NUM_SPLITS}: Selecting a new random sample...")
        (x, y), _ = get_adult_dataset_pd()
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, stratify=y, random_state=i)
        
        features = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        x_train, x_test = pd.DataFrame(x_train, columns=features), pd.DataFrame(x_test, columns=features)
        
        sensitive_attr = x_train['race'].values
        intersectional_attr = x_train['sex'].values  # Adding gender for intersectional fairness
        
        print(f"Unique gender values before preprocessing: {np.unique(intersectional_attr)}")
        
        categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                'hours-per-week', 'native-country']
        numeric_features = [f for f in features if f not in categorical_features]
        preprocessor, encoded = create_encoder(numeric_features, categorical_features, x_train)
        
        base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2, min_samples_leaf=1)
        model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
        model.fit(ArrayDataset(encoded, y_train))
        predictions = model.predict(ArrayDataset(preprocessor.transform(x_test)))
        if predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)
        
        metrics, di, intersectional_metrics = compute_fairness_metrics(y_test, predictions, x_test['race'].values, x_test['sex'].values)
        disparate_impact_scores.append(di)
    
    print(f"Disparate Impact: Mean = {np.mean(disparate_impact_scores):.4f}, Worst Case = {min(disparate_impact_scores):.4f}")
    
    if min(disparate_impact_scores) < 0.8:
        print("WARNING: Fairness results are unstable across dataset splits.")
    else:
        print("Fairness is stable across dataset splits.")



