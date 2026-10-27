import hashlib

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.scoring import score_transactions, validate_model_bundle


@pytest.fixture
def bundle():
    training = pd.DataFrame(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        columns=["V1", "Amount"],
    )
    scaler = StandardScaler().fit(training)
    model = GaussianMixture(
        n_components=1,
        covariance_type="diag",
        random_state=42,
        n_init=1,
        reg_covar=1e-3,
    ).fit(scaler.transform(training))
    return {
        "model_version": "1.0",
        "feature_names": ["V1", "Amount"],
        "scaler": scaler,
        "model": model,
        "selected_components": 1,
        "threshold": 5.0,
        "threshold_convention": "negative_log_likelihood_greater_than_or_equal",
        "target_false_positive_rate": 0.01,
        "feature_configuration": "without_time",
        "covariance_type": "diag",
        "gmm_parameters": {"covariance_type": "diag"},
        "random_seed": 42,
        "software_versions": {"scikit_learn": "test"},
        "cleaned_dataset_sha256": hashlib.sha256(b"test data").hexdigest(),
    }


def test_scores_anomalies_in_correct_direction_and_applies_threshold(bundle):
    transactions = pd.DataFrame(
        {"V1": [0.0, 10.0], "Amount": [0.0, 10.0]}
    )

    result = score_transactions(transactions, bundle)

    assert result.loc[0, "anomaly_score"] < result.loc[1, "anomaly_score"]
    assert result["is_anomaly"].tolist() == [False, True]
    np.testing.assert_allclose(
        result["anomaly_score"].to_numpy(),
        -result["log_likelihood"].to_numpy(),
    )


def test_threshold_comparison_flags_equal_and_above_but_not_below(bundle):
    transaction = pd.DataFrame({"V1": [4.0], "Amount": [4.0]})
    anomaly_score = -bundle["model"].score_samples(
        bundle["scaler"].transform(transaction)
    )[0]

    bundle["threshold"] = float(anomaly_score)
    assert score_transactions(transaction, bundle)["is_anomaly"].iloc[0]
    bundle["threshold"] = float(np.nextafter(anomaly_score, np.inf))
    assert not score_transactions(transaction, bundle)["is_anomaly"].iloc[0]
    bundle["threshold"] = float(np.nextafter(anomaly_score, -np.inf))
    assert score_transactions(transaction, bundle)["is_anomaly"].iloc[0]


def test_preserves_input_index_and_ignores_class_label(bundle):
    transactions = pd.DataFrame(
        {"V1": [0.0, 8.0], "Amount": [1.0, 2.0], "Class": [0, 1]},
        index=[12, 18],
    )

    result = score_transactions(transactions, bundle)

    assert result.index.tolist() == [12, 18]
    assert result["is_anomaly"].tolist() == [False, True]


def test_rejects_missing_feature(bundle):
    with pytest.raises(ValueError, match="missing required features"):
        score_transactions(pd.DataFrame({"V1": [1.0]}), bundle)


def test_rejects_features_in_wrong_order(bundle):
    transactions = pd.DataFrame({"Amount": [1.0], "V1": [1.0]})
    with pytest.raises(ValueError, match="saved order"):
        score_transactions(transactions, bundle)


def test_rejects_unexpected_column(bundle):
    transactions = pd.DataFrame(
        {"V1": [1.0], "Amount": [1.0], "transaction_id": [3]}
    )
    with pytest.raises(ValueError, match="unexpected columns"):
        score_transactions(transactions, bundle)


def test_rejects_duplicate_columns(bundle):
    transactions = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["V1", "V1", "Amount"])
    with pytest.raises(ValueError, match="duplicate columns"):
        score_transactions(transactions, bundle)


@pytest.mark.parametrize(
    "values, error",
    [
        (["not numeric"], TypeError),
        ([np.nan], ValueError),
        ([np.inf], ValueError),
        ([-np.inf], ValueError),
    ],
)
def test_rejects_non_numeric_or_non_finite_features(bundle, values, error):
    transactions = pd.DataFrame({"V1": values, "Amount": [1.0]})
    with pytest.raises(error):
        score_transactions(transactions, bundle)


def test_rejects_empty_input(bundle):
    transactions = pd.DataFrame(columns=["V1", "Amount"])
    with pytest.raises(ValueError, match="at least one row"):
        score_transactions(transactions, bundle)


def test_rejects_incomplete_or_incompatible_model_bundle(bundle):
    incomplete = dict(bundle)
    incomplete.pop("threshold")
    with pytest.raises(ValueError, match="threshold"):
        validate_model_bundle(incomplete)

    incompatible = dict(bundle)
    incompatible["feature_names"] = ["V1"]
    with pytest.raises(ValueError, match="expects 2 features"):
        validate_model_bundle(incompatible)

    target_leakage = dict(bundle)
    target_leakage["feature_names"] = ["V1", "Class"]
    with pytest.raises(ValueError, match="Class"):
        validate_model_bundle(target_leakage)


def test_inference_does_not_refit_scaler(bundle):
    scaler = bundle["scaler"]
    mean_before = scaler.mean_.copy()
    samples_before = scaler.n_samples_seen_

    score_transactions(pd.DataFrame({"V1": [0.0], "Amount": [1.0]}), bundle)

    np.testing.assert_array_equal(scaler.mean_, mean_before)
    assert scaler.n_samples_seen_ == samples_before


def test_saved_and_reloaded_bundle_produces_identical_scores(tmp_path, bundle):
    transactions = pd.DataFrame(
        {"V1": [-2.0, 0.0, 3.0], "Amount": [1.0, 2.0, 3.0]}
    )
    expected = score_transactions(transactions, bundle)
    artifact = tmp_path / "gmm_model.joblib"
    joblib.dump(bundle, artifact)

    actual = score_transactions(transactions, joblib.load(artifact))

    pd.testing.assert_frame_equal(expected, actual)


def test_fixed_seed_makes_small_gmm_fit_reproducible():
    rng = np.random.RandomState(12)
    training = np.concatenate(
        [rng.normal(-2, 0.3, size=(80, 2)), rng.normal(2, 0.3, size=(80, 2))]
    )
    first = GaussianMixture(n_components=2, random_state=42, n_init=1).fit(training)
    second = GaussianMixture(n_components=2, random_state=42, n_init=1).fit(training)

    np.testing.assert_allclose(first.score_samples(training), second.score_samples(training))
