"""
subgroups.py - model performance by hotel and by guest country.

The brief's ethics section asks for performance "broken down by whatever
subgroups your data allows". Our data has no age or gender, but it does
have the guest's country and the hotel, so we report those. Results are
computed on the TEST set from the predictions bhooshan_models.py saved,
for the main model (with country) and the comparison model (without).

For each group we report:
  - bookings and the real cancellation rate
  - mean predicted probability (is the model too high or too low here?)
  - PR-AUC (how well it ranks cancellations within the group)
  - false-alarm rate: share of guests who DID arrive but were flagged as
    "will cancel" at the cost-based threshold of 0.667 (walk = 2x empty
    room). These are the guests a hotel would risk walking, so this is the
    error that harms people, and the one to compare across countries.

Countries with fewer than 500 test bookings are grouped as "Other" so each
rate rests on enough bookings to mean something.

Owner: Nohur Muhammad Jamiil, for Angishaan Bissessur's ethics section.

Run:  python subgroups.py   (after bhooshan_models.py)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import config

THRESHOLD = 2 / (2 + 1)      # cost-based: walk = 2x an empty room
MIN_GROUP = 500
FILES = {
    "with country": "bhooshan_selected_test_predictions.csv",
    "without country": "bhooshan_selected_test_predictions_no_country.csv",
}


def group_table(pred, column):
    """Return one row of metrics per value of `column`."""
    y = pred[config.TARGET]
    p = pred["predicted_cancel_probability"]
    rows = []
    for name, idx in pred.groupby(column).groups.items():
        yy, pp = y[idx], p[idx]
        stayed = yy == 0
        rows.append({
            column: name,
            "bookings": len(yy),
            "actual_cancel_rate": yy.mean(),
            "mean_predicted": pp.mean(),
            "pr_auc": average_precision_score(yy, pp) if yy.nunique() > 1 else np.nan,
            "false_alarm_rate": (pp[stayed] >= THRESHOLD).mean() if stayed.any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values("bookings", ascending=False)


def run():
    tables = []
    for model_name, file in FILES.items():
        path = config.OUTPUT_DIR / file
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run bhooshan_models.py first.")
        pred = pd.read_csv(path)
        counts = pred["country"].value_counts()
        big = counts[counts >= MIN_GROUP].index
        pred["country_group"] = pred["country"].where(pred["country"].isin(big), "Other")

        print("=" * 78)
        print(f"SUBGROUPS ON THE TEST SET - model {model_name} "
              f"(false alarm = flagged at p >= {THRESHOLD:.3f} but guest arrived)")
        print("=" * 78)
        for column in ["hotel", "country_group"]:
            t = group_table(pred, column)
            print(t.round(3).to_string(index=False), "\n")
            tables.append(t.rename(columns={column: "group"})
                           .assign(model=model_name, grouping=column))

    out = config.OUTPUT_DIR / "subgroups.csv"
    pd.concat(tables, ignore_index=True).to_csv(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    run()
