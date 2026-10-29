"""
bhooshan_models.py - Section 4 and 5: the three models and their evaluation.

Owner: Bhooshan Sewock. Originally written in Google Colab; converted to a
plain script for submission (notebooks are not accepted by the brief). The
modelling logic is unchanged; integration changes are marked "INTEGRATION".

What it does, in order:
  1. Loads the team data and the saved split (data.py, split.py) and checks,
     by file fingerprint, that the split was not changed.
  2. Validation-only diagnostics: leakage check and duplicates kept/removed.
  3. Trains logistic regression, CatBoost and an embedding MLP, picks the best
     on validation average precision, then scores the held-out test set once.
  4. Repeats steps 3 without the country column, for the ethics comparison.
  5. Writes predictions, calibration bands and a JSON record to outputs/.

Run:  python bhooshan_models.py
"""

import hashlib
import json
from pathlib import Path
import inspect
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config
import data
import split

print("Imported data.py:", inspect.getfile(data))
print("Imported split.py:", inspect.getfile(split))


def sha256_file(path):
    """Return a fingerprint of an existing file without modifying it."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_team_data():
    """Use Jamiil's loader and verify his saved row-level split."""
    required_config = (
        "RAW_DATA_PATH", "SPLIT_FILE", "OUTPUT_DIR",
        "SPLIT_ORDER_COLUMN", "SEED", "TARGET",
    )
    missing = [name for name in required_config if not hasattr(config, name)]
    if missing:
        raise AttributeError(f"Missing config.py settings: {missing}")

    raw_path = Path(config.RAW_DATA_PATH)
    split_path = Path(config.SPLIT_FILE)

    if not raw_path.exists() or not split_path.exists():
        raise FileNotFoundError(
            f"Check configured paths:\nraw={raw_path}\nsplit={split_path}"
        )

    raw_hash = sha256_file(raw_path)
    split_hash = sha256_file(split_path)

    saved = pd.read_csv(split_path)
    required = {"row_id", "set", "ordered_by"}
    if not required.issubset(saved.columns):
        raise ValueError(f"split.csv needs {sorted(required)}")
    if saved.empty or saved["row_id"].duplicated().any():
        raise ValueError("split.csv is empty or has duplicate row IDs.")
    if saved["ordered_by"].nunique() != 1:
        raise ValueError("split.csv must record one ordering.")
    if saved["ordered_by"].iloc[0] != config.SPLIT_ORDER_COLUMN:
        raise ValueError(
            "split.csv ordering differs from config.py. Ask Jamiil."
        )

    loaded = data.load_data()

    # Use Jamiil's actual get_split interface, not a second split routine.
    if "df" in inspect.signature(split.get_split).parameters:
        frame, returned_labels = split.get_split(df=loaded)
    else:
        frame, returned_labels = split.get_split()

    if sha256_file(raw_path) != raw_hash:
        raise RuntimeError("Raw data changed during loading.")
    if sha256_file(split_path) != split_hash:
        raise RuntimeError("get_split changed split.csv; ask Jamiil why.")

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("get_split must return a DataFrame as its first value.")
    if not isinstance(returned_labels, pd.Series):
        raise TypeError("get_split must return a Series of split labels.")
    if not frame.index.is_unique or not returned_labels.index.is_unique:
        raise ValueError("Loaded row IDs or split-label IDs are duplicated.")

    saved_labels = saved.set_index("row_id")["set"]
    if len(frame) != len(saved_labels) or set(frame.index) != set(saved_labels.index):
        raise ValueError(
            "The uploaded split.csv does not match these data row IDs."
        )

    actual = returned_labels.reindex(frame.index)
    expected = saved_labels.reindex(frame.index)

    if actual.isna().any() or not np.array_equal(
        actual.to_numpy(), expected.to_numpy()
    ):
        raise ValueError(
            "get_split assignments differ from uploaded split.csv."
        )

    if set(actual.unique()) != {"train", "val", "test"}:
        raise ValueError("Expected train, val and test labels.")

    print("Seed:", config.SEED)
    print("Target:", config.TARGET)
    print("Ordering:", config.SPLIT_ORDER_COLUMN)
    print("Raw SHA-256:", raw_hash)
    print("Split SHA-256:", split_hash)
    print("Rows:", len(frame))
    print(actual.value_counts().to_string())

    return frame, actual, raw_hash, split_hash


random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.set_num_threads(2)

df, labels, DATA_HASH, SPLIT_HASH = load_team_data()

# INTEGRATION: the feature list is the team's final list from config.py, so
# every model and analysis uses identical columns. Previously a cautious
# 14-column list; parking is now excluded as a likely leak (see config.py).
NUMERIC, CATEGORICAL = config.model_features(include_country=True)

FEATURES = NUMERIC + CATEGORICAL


def set_feature_set(include_country):
    """INTEGRATION: switch the global feature lists for the with/without
    country comparison. All functions below read these globals at call time."""
    global NUMERIC, CATEGORICAL, FEATURES
    NUMERIC, CATEGORICAL = config.model_features(include_country=include_country)
    FEATURES = NUMERIC + CATEGORICAL

# Intentionally invalid experiment only.
DIRECT_OUTCOME_CATEGORICAL = ["reservation_status"]

# Investigate recording time; inclusion in this experiment does NOT prove
# that each field leaks.
DISPUTED_CATEGORICAL = ["assigned_room_type", "country"]
DISPUTED_NUMERIC = [
    "booking_changes",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "days_in_waiting_list",
    "adr",
]


def check_columns(frame):
    """Fail clearly if Jamiil's loaded schema differs from expectations."""
    required = set(
        FEATURES
        + DIRECT_OUTCOME_CATEGORICAL
        + DISPUTED_CATEGORICAL
        + DISPUTED_NUMERIC
        + ["reservation_status_date", "arrival_date", config.TARGET]
    )
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")


def category_strings(series):
    """Represent missing categories consistently."""
    return series.astype("string").fillna("__MISSING__").astype(str)


def prepare_features(rows, experiment="clean"):
    """Type feature values without fitting on evaluation rows."""
    if experiment not in {"clean", "direct_outcome", "all_flagged"}:
        raise ValueError(f"Unknown experiment: {experiment}")

    prepared = rows.copy()
    numeric = NUMERIC.copy()
    categorical = CATEGORICAL.copy()

    if experiment in {"direct_outcome", "all_flagged"}:
        categorical += DIRECT_OUTCOME_CATEGORICAL
        numeric += ["outcome_date_ordinal"]

        dates = pd.to_datetime(
            prepared["reservation_status_date"], errors="coerce"
        )
        prepared["outcome_date_ordinal"] = dates.map(
            lambda value: float(value.toordinal())
            if pd.notna(value) else np.nan
        )

    if experiment == "all_flagged":
        # INTEGRATION: several disputed fields are now regular features, so
        # only add the ones not already present (avoids duplicate columns).
        categorical += [c for c in DISPUTED_CATEGORICAL if c not in categorical]
        numeric += [c for c in DISPUTED_NUMERIC if c not in numeric]

    for column in numeric:
        prepared[column] = pd.to_numeric(
            prepared[column], errors="coerce"
        ).astype("float64")

    for column in categorical:
        prepared[column] = category_strings(prepared[column])

    return prepared, numeric, categorical


def one_hot_encoder():
    """Support scikit-learn versions used in different Colab runtimes."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def logistic_model(numeric, categorical):
    """Fit preprocessing and logistic regression on training data only."""
    preprocessing = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ),
        ("categorical", one_hot_encoder(), categorical),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=1500,
            random_state=config.SEED,
        )),
    ])


def scores(actual, probabilities):
    """Return average precision and Brier score."""
    return {
        "average_precision": float(
            average_precision_score(actual, probabilities)
        ),
        "brier": float(brier_score_loss(actual, probabilities)),
    }


def fit_logistic(training, validation, experiment="clean"):
    """Fit one logistic model and score the fixed validation rows."""
    x_train, numeric, categorical = prepare_features(
        training, experiment
    )
    x_val, _, _ = prepare_features(validation, experiment)
    columns = numeric + categorical

    model = logistic_model(numeric, categorical)
    model.fit(x_train[columns], training[config.TARGET])

    probabilities = model.predict_proba(x_val[columns])[:, 1]
    return model, probabilities, scores(
        validation[config.TARGET], probabilities
    )


check_columns(df)
print("Proposed numeric inputs:", NUMERIC)
print("Proposed categorical inputs:", CATEGORICAL)

def duplicate_columns(frame):
    """Choose source columns rather than engineered date columns.

    Confirm with Jamiil whether this matches the duplicate definition
    used for his reported 31,994 dataset-wide duplicate copies.
    """
    return [
        column for column in frame.columns
        if column not in {"arrival_date", "booking_date"}
    ]


def run_diagnostics(frame, split_labels):
    """Measure leakage and train-only deduplication on fixed validation."""
    training = frame.loc[split_labels == "train"].copy()
    validation = frame.loc[split_labels == "val"].copy()

    if training.empty or validation.empty:
        raise ValueError("Training and validation must be nonempty.")

    print("VALIDATION-ONLY DIAGNOSTICS")
    print("Training rows:", len(training))
    print("Validation rows:", len(validation))
    print(
        "Validation cancellation prevalence:",
        f"{validation[config.TARGET].mean():.4f}",
    )

    results = {}
    for experiment in ("all_flagged", "direct_outcome", "clean"):
        _, _, result = fit_logistic(
            training, validation, experiment
        )
        results[experiment] = result
        print(
            f"{experiment:>16}: "
            f"average precision={result['average_precision']:.4f}; "
            f"Brier={result['brier']:.4f}"
        )

    deduplicated = training.drop_duplicates(
        subset=duplicate_columns(frame),
        keep="first",
    )
    removed = len(training) - len(deduplicated)

    _, _, dedup_result = fit_logistic(
        deduplicated, validation, "clean"
    )
    results["clean_train_deduplicated"] = dedup_result
    results["training_duplicate_copies_removed"] = int(removed)

    print("Duplicate copies removed FROM TRAIN ONLY:", removed)
    print(
        "clean_train_deduplicated: "
        f"average precision={dedup_result['average_precision']:.4f}; "
        f"Brier={dedup_result['brier']:.4f}"
    )
    print("Held-out test was not evaluated.")

    return results, training, validation, deduplicated


diagnostics, train, val, train_deduplicated = run_diagnostics(
    df, labels
)

def fit_catboost(training, validation):
    """Fit gradient boosting with train-fitted numeric imputation."""
    x_train, _, _ = prepare_features(training, "clean")
    x_val, _, _ = prepare_features(validation, "clean")

    medians = x_train[NUMERIC].median().fillna(0.0)
    x_train[NUMERIC] = x_train[NUMERIC].fillna(medians)
    x_val[NUMERIC] = x_val[NUMERIC].fillna(medians)

    model = CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        random_seed=config.SEED,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        x_train[FEATURES],
        training[config.TARGET],
        cat_features=CATEGORICAL,
    )

    probabilities = model.predict_proba(x_val[FEATURES])[:, 1]
    fitted = {"model": model, "numeric_medians": medians}
    return fitted, probabilities, scores(
        validation[config.TARGET], probabilities
    )


class MLPConverter:
    """Learn numeric scaling and category vocabularies on train only."""

    def fit(self, training):
        """Fit transformations without seeing validation or test values."""
        frame, _, _ = prepare_features(training, "clean")
        numbers = frame[NUMERIC]

        self.medians = numbers.median().fillna(0.0)
        filled = numbers.fillna(self.medians)
        self.means = filled.mean()
        self.stds = filled.std(ddof=0).replace(0, 1.0).fillna(1.0)

        # ID 0 = missing; ID 1 = category not seen during training.
        self.vocabularies = {}
        for column in CATEGORICAL:
            known = sorted(
                value for value in frame[column].unique()
                if value != "__MISSING__"
            )
            self.vocabularies[column] = {
                value: index + 2
                for index, value in enumerate(known)
            }
        return self

    def transform(self, rows):
        """Apply only statistics and IDs learned from training."""
        frame, _, _ = prepare_features(rows, "clean")

        numeric = (
            frame[NUMERIC]
            .fillna(self.medians)
            .sub(self.means)
            .div(self.stds)
            .to_numpy(dtype=np.float32)
        )

        encoded_columns = []
        for column in CATEGORICAL:
            vocabulary = self.vocabularies[column]
            encoded_columns.append([
                0 if value == "__MISSING__"
                else vocabulary.get(value, 1)
                for value in frame[column]
            ])

        categorical = np.asarray(
            encoded_columns, dtype=np.int64
        ).T

        return (
            torch.from_numpy(numeric),
            torch.from_numpy(categorical),
        )

    def category_sizes(self):
        """Allow IDs for both missing and unseen categories."""
        return [
            len(self.vocabularies[column]) + 2
            for column in CATEGORICAL
        ]


class EmbeddingMLP(nn.Module):
    """Predict cancellation using numeric inputs and category embeddings."""

    def __init__(self, category_sizes, numeric_count):
        """Construct embeddings and a small two-hidden-layer network."""
        super().__init__()
        dimensions = [
            min(16, max(2, int(size ** 0.5)))
            for size in category_sizes
        ]
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dimension)
            for size, dimension in zip(category_sizes, dimensions)
        ])

        width = numeric_count + sum(dimensions)
        self.head = nn.Sequential(
            nn.Linear(width, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, numeric, categorical):
        """Return one cancellation logit per booking."""
        pieces = [numeric]
        for index, embedding in enumerate(self.embeddings):
            pieces.append(embedding(categorical[:, index]))
        return self.head(torch.cat(pieces, dim=1)).squeeze(1)


def mlp_probabilities(network, inputs):
    """Score batches without updating network weights."""
    network.eval()
    loader = DataLoader(
        TensorDataset(*inputs),
        batch_size=2048,
        shuffle=False,
    )
    batches = []

    with torch.no_grad():
        for numeric, categorical in loader:
            batches.append(
                torch.sigmoid(
                    network(numeric, categorical)
                ).cpu().numpy()
            )

    return np.concatenate(batches)


def fit_mlp(training, validation):
    """Train embeddings and select an epoch using validation AP only."""
    torch.manual_seed(config.SEED)

    converter = MLPConverter().fit(training)
    train_inputs = converter.transform(training)
    val_inputs = converter.transform(validation)

    targets = torch.tensor(
        training[config.TARGET].to_numpy(dtype=np.float32)
    )
    loader = DataLoader(
        TensorDataset(*train_inputs, targets),
        batch_size=512,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.SEED),
    )

    network = EmbeddingMLP(
        converter.category_sizes(), len(NUMERIC)
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    loss_function = nn.BCEWithLogitsLoss()

    best_ap = -1.0
    best_weights = None
    stale_epochs = 0

    for epoch in range(1, 21):
        network.train()

        for numeric, categorical, actual in loader:
            optimizer.zero_grad()
            loss = loss_function(
                network(numeric, categorical), actual
            )
            loss.backward()
            optimizer.step()

        val_p = mlp_probabilities(network, val_inputs)
        current_ap = average_precision_score(
            validation[config.TARGET], val_p
        )
        print(
            f"MLP epoch {epoch:02d}: "
            f"validation average precision={current_ap:.4f}"
        )

        if current_ap > best_ap + 0.0001:
            best_ap = current_ap
            best_weights = deepcopy(network.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 4:
                break

    if best_weights is None:
        raise RuntimeError("MLP did not produce a usable checkpoint.")

    network.load_state_dict(best_weights)
    val_p = mlp_probabilities(network, val_inputs)

    fitted = {"network": network, "converter": converter}
    return fitted, val_p, scores(
        validation[config.TARGET], val_p
    )


def predict_model(name, fitted, rows):
    """Apply a fitted legitimate model and its training preprocessing."""
    prepared, _, _ = prepare_features(rows, "clean")

    if name == "logistic":
        return fitted.predict_proba(prepared[FEATURES])[:, 1]

    if name == "catboost":
        prepared[NUMERIC] = prepared[NUMERIC].fillna(
            fitted["numeric_medians"]
        )
        return fitted["model"].predict_proba(
            prepared[FEATURES]
        )[:, 1]

    if name == "mlp":
        return mlp_probabilities(
            fitted["network"],
            fitted["converter"].transform(rows),
        )

    raise ValueError(f"Unknown model: {name}")

import json


def prediction_table(rows, probabilities, subset, model_name):
    """Provide row-ID-keyed predictions for teammates' analyses."""
    columns = [
        "hotel",
        "country",
        "arrival_date",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        config.TARGET,
    ]
    table = rows[columns].copy()
    table.insert(0, "row_id", rows.index)
    table["predicted_cancel_probability"] = probabilities
    table["set"] = subset
    table["model"] = model_name
    return table


def calibration_table(rows, probabilities, subset):
    """Compare average predicted risk and observed rates in risk bands."""
    values = pd.DataFrame({
        "actual": rows[config.TARGET].to_numpy(),
        "probability": probabilities,
    })
    values["band"] = pd.cut(
        values["probability"],
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
    )

    summary = (
        values.groupby("band", observed=True)
        .agg(
            bookings=("actual", "size"),
            mean_predicted=("probability", "mean"),
            observed_cancel_rate=("actual", "mean"),
        )
        .reset_index()
    )
    summary["band"] = summary["band"].astype(str)
    summary.insert(0, "set", subset)
    return summary


def run_final_models(
    *,
    include_country=True,
    duplicate_policy,
    approved_data_hash,
    approved_split_hash,
    booking_time_features_approved,
):
    """Choose on validation, then evaluate held-out test exactly afterward."""
    if duplicate_policy not in {"keep", "remove"}:
        raise ValueError("Arfi must decide 'keep' or 'remove'.")
    if not booking_time_features_approved:
        raise ValueError("The team must approve feature recording times.")
    if approved_data_hash != DATA_HASH:
        raise ValueError("Raw data hash is not the approved hash.")
    if approved_split_hash != SPLIT_HASH:
        raise ValueError("Split hash is not the approved hash.")
    if sha256_file(config.RAW_DATA_PATH) != DATA_HASH:
        raise ValueError("Raw data changed since the diagnostic run.")
    if sha256_file(config.SPLIT_FILE) != SPLIT_HASH:
        raise ValueError("Split changed since the diagnostic run.")

    set_feature_set(include_country)
    suffix = "" if include_country else "_no_country"
    print(f"\n===== FEATURE SET: {'with' if include_country else 'WITHOUT'} country =====")

    training = (
        train if duplicate_policy == "keep"
        else train_deduplicated
    )

    trainers = {
        "logistic": fit_logistic,
        "catboost": fit_catboost,
        "mlp": fit_mlp,
    }
    fitted_models = {}
    val_predictions = {}
    val_scores = {}

    for name, trainer in trainers.items():
        fitted, probabilities, result = trainer(training, val)
        fitted_models[name] = fitted
        val_predictions[name] = probabilities
        val_scores[name] = result

        print(
            f"VALIDATION {name}: "
            f"average precision={result['average_precision']:.4f}; "
            f"Brier={result['brier']:.4f}"
        )

    # Fixed selection rule: highest validation average precision;
    # lower Brier breaks a tie. Do not select using test results.
    selected = max(
        val_scores,
        key=lambda name: (
            val_scores[name]["average_precision"],
            -val_scores[name]["brier"],
        ),
    )
    print("Selected using validation:", selected)

    test = df.loc[labels == "test"].copy()
    # INTEGRATION: the team reports every test score twice: on all test
    # bookings, and on test bookings that are not copies of an earlier row
    # (the same definition Arfi uses), because group blocks flatter the score.
    is_copy = df.duplicated(subset=duplicate_columns(df), keep="first")
    test_unique = ~is_copy.loc[test.index].to_numpy()
    test_scores = {}
    test_scores_dedup = {}
    selected_test_predictions = None

    for name, fitted in fitted_models.items():
        probabilities = predict_model(name, fitted, test)
        result = scores(test[config.TARGET], probabilities)
        test_scores[name] = result
        test_scores_dedup[name] = scores(
            test[config.TARGET].to_numpy()[test_unique],
            probabilities[test_unique],
        )

        print(
            f"HELD-OUT TEST {name}: "
            f"average precision={result['average_precision']:.4f}; "
            f"Brier={result['brier']:.4f}   |   de-duplicated test: "
            f"average precision={test_scores_dedup[name]['average_precision']:.4f}"
        )

        if name == selected:
            selected_test_predictions = probabilities

    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_table(
        val, val_predictions[selected], "val", selected
    ).to_csv(
        output_dir / f"bhooshan_selected_val_predictions{suffix}.csv",
        index=False,
    )

    prediction_table(
        test, selected_test_predictions, "test", selected
    ).to_csv(
        output_dir / f"bhooshan_selected_test_predictions{suffix}.csv",
        index=False,
    )

    pd.concat([
        calibration_table(val, val_predictions[selected], "val"),
        calibration_table(
            test, selected_test_predictions, "test"
        ),
    ], ignore_index=True).to_csv(
        output_dir / f"bhooshan_calibration_bands{suffix}.csv",
        index=False,
    )

    record = {
        "seed": int(config.SEED),
        "split_ordered_by": config.SPLIT_ORDER_COLUMN,
        "raw_data_sha256": DATA_HASH,
        "split_sha256": SPLIT_HASH,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "duplicate_policy": duplicate_policy,
        "training_rows": len(training),
        "validation_rows": len(val),
        "test_rows": len(test),
        "selected_model": selected,
        "diagnostics": diagnostics,
        "validation_scores": val_scores,
        "include_country": include_country,
        "test_scores": test_scores,
        "test_scores_deduplicated": test_scores_dedup,
        "test_rows_deduplicated": int(test_unique.sum()),
    }

    with open(
        output_dir / f"bhooshan_results{suffix}.json",
        "w",
        encoding="utf-8",
    ) as destination:
        json.dump(record, destination, indent=2)

    # INTEGRATION: raw (uncalibrated) probabilities of the selected model for
    # every validation and test row, in the format Arfi's overbooking code
    # reads. Only the main (with-country) run writes this file.
    if include_country:
        # INTEGRATION: save the selected model for the demo (predict.py).
        # A calibrator is fitted on validation only, the same method Arfi
        # uses, so the demo shows calibrated percentages. Only the
        # logistic and CatBoost models are saved this way; the MLP depends on
        # classes defined in this script, which would rerun it on import.
        if selected in ("logistic", "catboost"):
            import joblib
            from sklearn.isotonic import IsotonicRegression
            calibrator = IsotonicRegression(out_of_bounds="clip").fit(
                val_predictions[selected], val[config.TARGET])
            joblib.dump({
                "name": selected,
                "fitted": fitted_models[selected],
                "numeric": list(NUMERIC),
                "categorical": list(CATEGORICAL),
                "calibrator": calibrator,
            }, output_dir / "final_model.joblib")
            print("Final model saved to:", output_dir / "final_model.joblib")
        else:
            print("Selected model is the MLP; final_model.joblib not written.")

        pd.DataFrame({
            "row_id": list(val.index) + list(test.index),
            "prob": list(val_predictions[selected])
                    + list(selected_test_predictions),
        }).to_csv(output_dir / "predictions.csv", index=False)

    print("Results written to:", output_dir)

    return {
        "results": record,
        "selected_model": selected,
        "selected_fitted_model": fitted_models[selected],
    }

DUPLICATE_POLICY = "keep"

if DUPLICATE_POLICY not in {"keep", "remove"}:
    raise ValueError("Enter Arfi's approved duplicate policy.")

print("Approved duplicate policy:", DUPLICATE_POLICY)
print("Dataset SHA-256:", DATA_HASH)
print("Split SHA-256:", SPLIT_HASH)
print("This run will evaluate the held-out test set.")

# INTEGRATION: main run with country, then the same pipeline without it for
# the ethics comparison. The test set is scored once per feature set.
final_run = run_final_models(
    include_country=True,
    duplicate_policy=DUPLICATE_POLICY,
    approved_data_hash=DATA_HASH,
    approved_split_hash=SPLIT_HASH,
    booking_time_features_approved=True,
)
no_country_run = run_final_models(
    include_country=False,
    duplicate_policy=DUPLICATE_POLICY,
    approved_data_hash=DATA_HASH,
    approved_split_hash=SPLIT_HASH,
    booking_time_features_approved=True,
)

for label, run in (("WITH country", final_run), ("WITHOUT country", no_country_run)):
    print(f"\n{label}: selected on validation = {run['selected_model']}")
    print("  Validation:", run["results"]["validation_scores"])
    print("  Test (all):", run["results"]["test_scores"])
    print("  Test (de-duplicated):", run["results"]["test_scores_deduplicated"])
