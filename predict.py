"""
predict.py - score new bookings with the team's final model.

This is the piece the demo (and a real hotel system) would call. It loads
outputs/final_model.joblib, written by bhooshan_models.py, and returns a
calibrated cancellation probability for each booking.

The preparation below repeats exactly what bhooshan_models.py does before
training (numbers as floats, categories as text with missing values given
their own label, CatBoost gaps filled with training medians), so a booking is
treated the same way at prediction time as during training. If the two ever
differed, the model would see inputs it was not trained on.

Owner: Nohur Muhammad Jamiil (integration).

Run:  python predict.py   (scores three test bookings as a check)
"""

import joblib
import pandas as pd

import config

MODEL_FILE = config.OUTPUT_DIR / "final_model.joblib"


def load_model(path=MODEL_FILE):
    """Load the saved model bundle (model, feature lists, calibrator)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run bhooshan_models.py (or main.py) first.")
    return joblib.load(path)


def prepare(bookings, bundle):
    """Turn raw booking rows into the model's input columns."""
    numeric, categorical = bundle["numeric"], bundle["categorical"]
    X = bookings[numeric + categorical].copy()
    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")
    for col in categorical:
        # Same rule as training: a missing agent/company/country is its own
        # category ("__MISSING__"), not a reason to drop the booking.
        X[col] = X[col].astype("string").fillna("__MISSING__").astype(str)
    if bundle["name"] == "catboost":
        X[numeric] = X[numeric].fillna(bundle["fitted"]["numeric_medians"])
    return X


def predict_proba(bookings, bundle=None):
    """Return calibrated cancellation probabilities, one per booking row."""
    bundle = bundle or load_model()
    X = prepare(bookings, bundle)
    model = (bundle["fitted"]["model"] if bundle["name"] == "catboost"
             else bundle["fitted"])
    raw = model.predict_proba(X)[:, 1]
    return bundle["calibrator"].predict(raw)


if __name__ == "__main__":
    from split import get_split

    df, labels = get_split()
    bundle = load_model()
    sample = df[labels == "test"].head(3)
    sample = sample.assign(predicted=predict_proba(sample, bundle))
    print(f"Model: {bundle['name']}")
    print(sample[["hotel", "lead_time", "deposit_type", "market_segment",
                  "predicted", config.TARGET]].round(3).to_string())
