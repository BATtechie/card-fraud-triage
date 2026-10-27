"""Validated inference helpers for saved GMM anomaly-detection bundles."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def score_transactions(transactions: pd.DataFrame, model_bundle: Dict[str, Any]) -> pd.DataFrame:
    """Score feature rows with a fitted bundle without refitting any preprocessing.

    An optional ``Class`` column is ignored for offline evaluation. Other extra
    columns are rejected to prevent accidental identifiers or target leakage.
    Feature columns must appear in the saved order.
    """
    if not isinstance(transactions, pd.DataFrame):
        raise TypeError("transactions must be a pandas DataFrame")
    if transactions.empty:
        raise ValueError("transactions must contain at least one row")
    if not isinstance(model_bundle, dict):
        raise TypeError("model_bundle must be a dictionary")

    feature_names, scaler, model, threshold = validate_model_bundle(model_bundle)
    if transactions.columns.duplicated().any():
        duplicates = transactions.columns[transactions.columns.duplicated()].tolist()
        raise ValueError("transactions contain duplicate columns: {}".format(duplicates))

    supplied_columns = [column for column in transactions.columns if column != "Class"]
    unexpected_columns = [column for column in supplied_columns if column not in feature_names]
    missing_columns = [column for column in feature_names if column not in supplied_columns]
    if missing_columns:
        raise ValueError("transactions are missing required features: {}".format(missing_columns))
    if unexpected_columns:
        raise ValueError("transactions contain unexpected columns: {}".format(unexpected_columns))
    if supplied_columns != feature_names:
        raise ValueError("transactions features must appear in the saved order: {}".format(feature_names))

    features = transactions.loc[:, feature_names]
    non_numeric = [
        column for column in feature_names
        if not pd.api.types.is_numeric_dtype(features[column])
    ]
    if non_numeric:
        raise TypeError("transaction features must be numeric: {}".format(non_numeric))
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("transaction features must not contain NaN or infinite values")

    transformed = np.asarray(scaler.transform(features), dtype=float)
    if transformed.shape != values.shape or not np.isfinite(transformed).all():
        raise ValueError("scaler returned invalid transformed features")
    log_likelihood = np.asarray(model.score_samples(transformed), dtype=float)
    if log_likelihood.shape != (len(transactions),) or not np.isfinite(log_likelihood).all():
        raise ValueError("GMM returned invalid log-likelihood scores")
    anomaly_score = -log_likelihood
    result = pd.DataFrame(
        {
            "log_likelihood": log_likelihood,
            "anomaly_score": anomaly_score,
            "is_anomaly": anomaly_score >= threshold,
        },
        index=transactions.index,
    )
    return result


def validate_model_bundle(model_bundle: Dict[str, Any]) -> Tuple[List[str], Any, Any, float]:
    """Raise a clear error unless a bundle is complete and dimensionally compatible."""
    feature_names = model_bundle.get("feature_names")
    scaler = model_bundle.get("scaler")
    model = model_bundle.get("model")
    threshold = model_bundle.get("threshold")
    bundle_version = model_bundle.get("model_version")
    if bundle_version != "1.0":
        raise ValueError("model_bundle has an unsupported model_version: {!r}".format(bundle_version))
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("model_bundle must contain a non-empty feature_names list")
    if any(not isinstance(name, str) for name in feature_names):
        raise ValueError("model_bundle feature_names must be unique strings")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("model_bundle feature_names must be unique strings")
    if "Class" in feature_names:
        raise ValueError("model_bundle must not include the Class label as a feature")
    if scaler is None or not callable(getattr(scaler, "transform", None)):
        raise ValueError("model_bundle must contain a fitted scaler with transform()")
    if model is None or not callable(getattr(model, "score_samples", None)):
        raise ValueError("model_bundle must contain a fitted GMM with score_samples()")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise ValueError("model_bundle threshold must be numeric")
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("model_bundle threshold must be finite")
    if model_bundle.get("threshold_convention") != "negative_log_likelihood_greater_than_or_equal":
        raise ValueError("model_bundle has an unsupported threshold convention")
    target_fpr = model_bundle.get("target_false_positive_rate")
    if isinstance(target_fpr, bool) or not isinstance(target_fpr, (int, float, np.integer, np.floating)):
        raise ValueError("model_bundle target_false_positive_rate must be numeric")
    if not 0 < float(target_fpr) < 1:
        raise ValueError("model_bundle target_false_positive_rate must be between 0 and 1")
    if not isinstance(model_bundle.get("feature_configuration"), str):
        raise ValueError("model_bundle must include feature_configuration")
    covariance_type = model_bundle.get("covariance_type")
    if not isinstance(covariance_type, str):
        raise ValueError("model_bundle must include covariance_type")
    if not isinstance(model_bundle.get("software_versions"), dict):
        raise ValueError("model_bundle must include software_versions")
    dataset_hash = model_bundle.get("cleaned_dataset_sha256")
    if not isinstance(dataset_hash, str) or len(dataset_hash) != 64:
        raise ValueError("model_bundle must include a SHA-256 hash of the cleaned dataset")
    try:
        int(dataset_hash, 16)
    except ValueError as error:
        raise ValueError("model_bundle cleaned dataset hash must be hexadecimal") from error
    selected_components = model_bundle.get("selected_components")
    if isinstance(selected_components, bool) or not isinstance(selected_components, (int, np.integer)) or selected_components < 1:
        raise ValueError("model_bundle selected_components must be a positive integer")
    if not isinstance(model_bundle.get("gmm_parameters"), dict):
        raise ValueError("model_bundle must include GMM parameters")
    if not isinstance(model_bundle.get("random_seed"), int):
        raise ValueError("model_bundle must include an integer random_seed")

    for estimator_name, estimator in (("scaler", scaler), ("model", model)):
        n_features = getattr(estimator, "n_features_in_", None)
        if n_features is None:
            raise ValueError("model_bundle {} is not fitted or lacks n_features_in_".format(estimator_name))
        if int(n_features) != len(feature_names):
            raise ValueError(
                "model_bundle {} expects {} features, but feature_names contains {}".format(
                    estimator_name, n_features, len(feature_names)
                )
            )
        fitted_names = getattr(estimator, "feature_names_in_", None)
        if fitted_names is not None and list(fitted_names) != feature_names:
            raise ValueError("model_bundle {} feature order does not match feature_names".format(estimator_name))

    weights = getattr(model, "weights_", None)
    if weights is None or not hasattr(scaler, "mean_"):
        raise ValueError("model_bundle scaler and GMM must be fitted")
    if int(selected_components) != len(weights):
        raise ValueError("model_bundle selected_components does not match the fitted GMM")
    model_components = getattr(model, "n_components", selected_components)
    if int(model_components) != int(selected_components):
        raise ValueError("model_bundle selected_components does not match the GMM configuration")
    if getattr(model, "covariance_type", covariance_type) != covariance_type:
        raise ValueError("model_bundle covariance_type does not match the fitted GMM")
    means = getattr(model, "means_", None)
    if means is None or np.asarray(means).shape != (int(selected_components), len(feature_names)):
        raise ValueError("model_bundle GMM means are incompatible with its component and feature counts")
    return feature_names, scaler, model, threshold
