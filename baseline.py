"""
baseline.py - a quick logistic regression used for diagnostic checks.

This is NOT the final baseline model. It is a small, fast, fixed recipe used
to answer "what happens to the score if we change X?" questions: removing
leaking columns, removing duplicate rows, changing the split ordering.
Keeping the recipe identical across those checks means any change in score
comes from the thing we changed, not from the model.

Every check is scored on the VALIDATION set. The test set stays untouched
until the final evaluation, as the brief requires.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def add_leak_features(df):
    """Turn reservation_status_date into a number so the model can use it.

    status_days = days between the status date and the arrival date. For a
    guest who stayed, it is the length of stay; for a cancellation it is
    negative (cancelled before arrival). Only used to demonstrate leakage.
    """
    df = df.copy()
    status_date = pd.to_datetime(df["reservation_status_date"])
    df["status_days"] = (status_date - df["arrival_date"]).dt.days
    return df


def quick_baseline(train, val, numeric, categorical):
    """Fit logistic regression on train, return PR-AUC and ROC-AUC on val.

    PR-AUC is our main metric (from the proposal): it focuses on the
    cancelled class, which is the minority. ROC-AUC is printed alongside
    because it is the number most public notebooks report.
    """
    prep = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        # handle_unknown="ignore": a category seen only in val (e.g. a new
        # country) is treated as all-zeros instead of crashing.
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = Pipeline([
        ("prep", prep),
        ("lr", LogisticRegression(max_iter=2000, random_state=config.SEED)),
    ])
    X_cols = numeric + categorical
    model.fit(train[X_cols].fillna(0), train[config.TARGET])
    p = model.predict_proba(val[X_cols].fillna(0))[:, 1]
    return {
        "pr_auc": average_precision_score(val[config.TARGET], p),
        "roc_auc": roc_auc_score(val[config.TARGET], p),
    }
