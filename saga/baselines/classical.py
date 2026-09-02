"""
RF-300, Logistic Regression, and the raw-SyncNet-threshold baselines
(Sections 3.4-3.5), evaluated under the closed-set and open-set protocols
used throughout the paper.

`X` is always the 36-d flat mean/std/max feature vector from
`saga.evaluation.features.extract_flat_xy`, optionally concatenated with the
2-d SyncNet scalars via `add_sync_features` for a 38-d input.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

DEFAULT_SEEDS = (42, 123, 777)


def fit_predict(clf, x_train, y_train, x_test):
    scaler = StandardScaler().fit(x_train)
    clf.fit(scaler.transform(x_train), y_train)
    return clf.predict(scaler.transform(x_test)), clf.predict_proba(scaler.transform(x_test))


def rf300(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                   class_weight="balanced", random_state=random_state, n_jobs=-1)


def logistic_regression(random_state: int) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)


def closed_set_multiseed(x_train, y_train, x_test, y_test, make_clf, seeds=DEFAULT_SEEDS) -> dict:
    """
    Random-split / identity-disjoint protocol: 4-class accuracy, macro F1,
    and binary (real-vs-fake) AUC, averaged over `seeds`.
    """
    bin_true = (y_test != 0).astype(int)
    accs, f1s, aucs = [], [], []
    for seed in seeds:
        preds, proba = fit_predict(make_clf(seed), x_train, y_train, x_test)
        accs.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds, average="macro", zero_division=0))
        aucs.append(roc_auc_score(bin_true, 1 - proba[:, 0]))
    return {
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
    }


def open_set_multiseed(x_train, y_train, x_test, y_test, make_clf,
                        held_out_label: int, seeds=DEFAULT_SEEDS) -> dict:
    """
    Open-set protocol: trains on the known classes only (`y_train` must
    already exclude `held_out_label`); `x_test`/`y_test` contain both known
    and held-out clips. Reports closed-set AUC among known classes (scored
    by 1 - max-softmax confidence) and open-set AUROC separating known vs.
    held-out (scored by max-softmax confidence — lower is more anomalous).
    """
    known_mask = y_test != held_out_label
    closed_aucs, open_aurocs = [], []
    for seed in seeds:
        _, proba = fit_predict(make_clf(seed), x_train, y_train, x_test)
        max_prob = proba.max(axis=1)

        known_bin = (y_test[known_mask] != 0).astype(int)
        known_conf = 1.0 - max_prob[known_mask]
        closed_aucs.append(roc_auc_score(known_bin, known_conf))

        binary_known = (y_test != held_out_label).astype(int)
        open_aurocs.append(roc_auc_score(binary_known, max_prob))
    return {
        "closed_auc_mean": float(np.mean(closed_aucs)), "closed_auc_std": float(np.std(closed_aucs)),
        "openset_auroc_mean": float(np.mean(open_aurocs)), "openset_auroc_std": float(np.std(open_aurocs)),
    }


def raw_syncnet_threshold_auc(sync_confidence: np.ndarray, y_test: np.ndarray) -> dict:
    """
    The SyncNet confidence scalar alone, no engineered features, no trained
    classifier — used directly as a real-vs-fake ranking score, exactly as
    the tool would be used out of the box. Reports both polarities honestly
    rather than assuming the naive one, since a sync-adversarially-trained
    generator (e.g. Wav2Lip) can push confidence in either direction.
    """
    bin_true = (y_test != 0).astype(int)
    auc_low_is_fake = roc_auc_score(bin_true, -sync_confidence)
    auc_high_is_fake = roc_auc_score(bin_true, sync_confidence)
    return {
        "auc_low_conf_is_fake": float(auc_low_is_fake),
        "auc_high_conf_is_fake": float(auc_high_is_fake),
        "best_auc": float(max(auc_low_is_fake, auc_high_is_fake)),
    }
