"""
arfi_overbooking.py - from cancellation probabilities to an overbooking decision.

Owner: Mohammad Arfi Ilshad Dolip.

What it does, in order:
  1. Gets probabilities for validation and test bookings: Bhooshan's selected
     model (outputs/predictions.csv), calibrated with isotonic regression
     fitted on validation only. A stand-in model is used only if that file
     is missing.
  2. Calibration report on the test set and outputs/reliability.png.
  3. Cost-based thresholds for several walk-cost assumptions.
  4. Nightly overbooking backtest on the test period, comparing the model with
     a fixed 10% rule, an average-rate rule and an oracle.
     Saves outputs/overbooking_cost.png.
  5. Error analysis on real test bookings.

Integration changes (team leader) are marked "INTEGRATION".

Run:  python arfi_overbooking.py   (after bhooshan_models.py)
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")   
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

import config
from arfi_duplicates import original_columns, prepare
from split import get_split

PREDICTIONS_FILE = config.OUTPUT_DIR / "predictions.csv"

# ---------------------------------------------------------------------------
# Cost assumptions (Section 1 of the report explains the sources)
# ---------------------------------------------------------------------------
# ce = cost of an empty room. Set to 1 so every cost is "in empty rooms".
# cw = cost of walking a guest, as a multiple of ce. A common hotel rule of thumb is 2x; a worked industry example uses 3x. We cannot observe the true figure for any one hotel, so we carry a range and show the decision under each.
CE = 1.0
WALK_MULTIPLES = [1, 2, 3, 5]
PRIMARY_MULTIPLE = 2          # the headline scenario in the report
TARGET_PRECISION = 0.90       # precision-first threshold, set on validation
FIXED_RULE_SHARE = 0.10       # the "overbook by 10%" rule from the proposal
N_SIMS = 2000                 # Monte Carlo draws per hotel night


# ---------------------------------------------------------------------------
# 1. Probabilities
# ---------------------------------------------------------------------------
def standin_probabilities(df, labels):
    """Train a gradient-boosting model on train, calibrate on validation.

    Returns a Series of probabilities for validation and test rows (NaN for
    train). Isotonic regression is fitted on validation only, so the test set
    is never used to tune anything. Categories seen fewer than 30 times are
    pooled so the encoder stays inside the trees' category limit.
    """
    X = prepare(df)
    cats = config.LOW_CARDINALITY_CATEGORICAL + config.HIGH_CARDINALITY_CATEGORICAL
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                min_frequency=30), cats)],
        remainder="passthrough")
    model = Pipeline([
        ("prep", pre),
        ("gb", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            categorical_features=list(range(len(cats))),
            random_state=config.SEED)),
    ])
    tr = labels == "train"
    va = labels == "val"
    te = labels == "test"
    model.fit(X[tr], df.loc[tr, config.TARGET])

    raw_val = model.predict_proba(X[va])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_val, df.loc[va, config.TARGET])
    prob = pd.Series(np.nan, index=df.index, name="prob")
    prob[va] = iso.predict(raw_val)
    prob[te] = iso.predict(model.predict_proba(X[te])[:, 1])
    raw = pd.Series(np.nan, index=df.index, name="raw")
    raw[va] = raw_val
    raw[te] = model.predict_proba(X[te])[:, 1]
    return prob, raw


def get_probabilities(df, labels):
    """Use the team's predictions if present, else the stand-in model.

    Returns (prob, raw, source). raw is the uncalibrated score where we have it.
    """
    if PREDICTIONS_FILE.exists():
        saved = pd.read_csv(PREDICTIONS_FILE, index_col="row_id")["prob"]
        raw = saved.reindex(df.index)
        raw[labels == "train"] = np.nan
        # INTEGRATION: the team model's probabilities arrive uncalibrated.
        # Calibrate them exactly as for the stand-in: isotonic regression
        # fitted on validation only, then applied to validation and test.
        va, te = labels == "val", labels == "test"
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            raw[va], df.loc[va, config.TARGET])
        prob = pd.Series(np.nan, index=df.index, name="prob")
        prob[va] = iso.predict(raw[va])
        prob[te] = iso.predict(raw[te])
        return prob, raw, "team model (outputs/predictions.csv), calibrated on validation"
    prob, raw = standin_probabilities(df, labels)
    return prob, raw, "STAND-IN gradient boosting (replace with team model)"


# ---------------------------------------------------------------------------
# 2. Calibration
# ---------------------------------------------------------------------------
def calibration_report(df, labels, prob, raw):
    """Brier score, PR-AUC and a reliability table on the test set."""
    te = labels == "test"
    y = df.loc[te, config.TARGET].to_numpy()
    print("=" * 64)
    print("CALIBRATION (test set)")
    print("=" * 64)
    print(f"Test bookings: {len(y):,}   cancel rate {y.mean():.1%}")
    print(f"PR-AUC {average_precision_score(y, prob[te]):.3f}   "
          f"Brier score {brier_score_loss(y, prob[te]):.3f}   "
          f"(always predicting the base rate: {brier_score_loss(y, np.full(len(y), y.mean())):.3f})")
    if raw.notna().any() and not np.allclose(raw[te], prob[te]):
        print(f"Brier before calibration {brier_score_loss(y, raw[te]):.3f} "
              f"-> after {brier_score_loss(y, prob[te]):.3f}")

    # Ten equal-width bins: if calibrated, mean prediction ~ actual cancel rate.
    bins = pd.cut(prob[te], np.linspace(0, 1, 11), include_lowest=True)
    tab = pd.DataFrame({"p": prob[te], "y": y}).groupby(bins, observed=True).agg(
        bookings=("y", "size"), mean_predicted=("p", "mean"), actual_rate=("y", "mean"))
    print(tab.round(3).to_string())
    # Whole-test check that matters for summing: expected vs actual cancellations.
    print(f"Sum of probabilities {prob[te].sum():,.0f} vs actual cancellations {int(y.sum()):,}")

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(tab["mean_predicted"], tab["actual_rate"], "o-", color="#3b6ea5", label="model")
    ax.set_xlabel("Predicted probability of cancelling")
    ax.set_ylabel("Share that actually cancelled")
    ax.set_title("Reliability on the test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "reliability.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Thresholds from costs
# ---------------------------------------------------------------------------
def cost_threshold(multiple):
    """Probability above which 'will cancel' is the cheaper call.

    Predicting a cancellation and overbooking on it gains ce if right and
    costs cw if the guest turns up, so call it when p*ce > (1-p)*cw, i.e.
    p > cw / (cw + ce). A dearer walk pushes the threshold up, which is
    exactly "favour precision on the cancelled class".
    """
    return multiple * CE / (multiple * CE + CE)


def threshold_report(df, labels, prob):
    """Compare cost-implied thresholds with the validation precision-first one."""
    va, te = labels == "val", labels == "test"
    yv, pv = df.loc[va, config.TARGET].to_numpy(), prob[va].to_numpy()
    yt, pt = df.loc[te, config.TARGET].to_numpy(), prob[te].to_numpy()

    prec, _, thr = precision_recall_curve(yv, pv)
    ok = np.where(prec[:-1] >= TARGET_PRECISION)[0]
    t_val = float(thr[ok[0]]) if len(ok) else float(thr[-1])

    rows = {}
    cases = [(f"cost-based, walk = {m}x empty room", cost_threshold(m)) for m in WALK_MULTIPLES]
    cases.append((f"validation, {TARGET_PRECISION:.0%} precision", t_val))
    for name, t in cases:
        pred = pt >= t
        tp = int((pred & (yt == 1)).sum())
        fp = int((pred & (yt == 0)).sum())
        rows[name] = {"threshold": t, "flagged": int(pred.sum()),
                      "precision": tp / max(int(pred.sum()), 1),
                      "recall": tp / int(yt.sum()),
                      "walk-risk (false alarms)": fp,
                      "missed cancels": int(yt.sum()) - tp}
    print("\n" + "=" * 64)
    print("THRESHOLDS (test set)")
    print("=" * 64)
    print(pd.DataFrame(rows).T.round(3).to_string())
    return cost_threshold(PRIMARY_MULTIPLE)


# ---------------------------------------------------------------------------
# 4. Nightly overbooking estimate
# ---------------------------------------------------------------------------
def build_nights(df, labels, prob):
    """One row per booking per night stayed, for validation and test bookings.

    Validation bookings are included only because stays that began just before
    the test period still occupy test nights. Only test-period nights are
    scored later. Zero-night bookings (a data oddity, flagged in data.py) are
    skipped because they occupy no night.
    """
    sel = labels.isin(["val", "test"]) & (df["stays_in_weekend_nights"]
                                          + df["stays_in_week_nights"] >= 1)
    b = df[sel].copy()
    b["prob"] = prob[sel]
    b["nights"] = b["stays_in_weekend_nights"] + b["stays_in_week_nights"]
    # Identical-row groups (see arfi_duplicates) are the unit that cancels together.
    b["unit"] = df.groupby(original_columns(df), dropna=False).ngroup()[sel]
    rep = b.loc[b.index.repeat(b["nights"])].copy()
    rep["night"] = rep["arrival_date"] + pd.to_timedelta(
        rep.groupby(level=0).cumcount(), unit="D")
    test_start = df.loc[labels == "test", "arrival_date"].min()
    test_end = df.loc[labels == "test", "arrival_date"].max()
    rep = rep[(rep["night"] >= test_start) & (rep["night"] <= test_end)]
    return rep[["hotel", "night", "unit", "prob", config.TARGET]]


def simulate_cancellations(p, units, rng):
    """Simulated cancellation counts for one hotel night, two ways.

    independent: every booking cancels on its own coin flip. This is the plain
        'add up the probabilities' view.
    block-aware: bookings that are identical rows are one reservation covering
        several rooms, so they cancel together (98.8% of identical rows share an
        outcome). One coin flip per block. Same average, wider spread.
    """
    ind = (rng.random((N_SIMS, len(p))) < p).sum(axis=1)
    ids, inv = np.unique(units, return_inverse=True)
    size = np.bincount(inv)
    p_unit = np.bincount(inv, weights=p) / size
    blk = (rng.random((N_SIMS, len(ids))) < p_unit) @ size
    return ind, blk


def overbooking_backtest(df, labels, prob):
    """Backtest overbooking rules on the test-period nights.

    For a night with B bookings of which X cancel, a hotel that overbooks by k
    rooms walks max(k - X, 0) guests and leaves max(X - k, 0) rooms empty. The
    cost-optimal k is the smallest value whose chance of being at least X is
    ce / (ce + cw): a lower quantile of the cancellation count, so it is
    cautious. We compare it with fixed rules and an oracle (k = X).
    """
    nights = build_nights(df, labels, prob)
    base_rate = df.loc[labels == "train", config.TARGET].mean()
    rows = []
    for (hotel, night), g in nights.groupby(["hotel", "night"]):
        rng = np.random.default_rng([config.SEED, int(night.value // 86_400_000_000_000),
                                     int(hotel == "City Hotel")])
        ind, blk = simulate_cancellations(g["prob"].to_numpy(), g["unit"].to_numpy(), rng)
        rec = {"hotel": hotel, "night": night, "booked": len(g),
               "actual": int(g[config.TARGET].sum()),
               "expected": g["prob"].sum(),
               "ind_lo": np.quantile(ind, 0.10), "ind_hi": np.quantile(ind, 0.90),
               "blk_lo": np.quantile(blk, 0.10), "blk_hi": np.quantile(blk, 0.90),
               "ind_sd": ind.std(), "blk_sd": blk.std(),
               "fixed10": int(FIXED_RULE_SHARE * len(g)),
               "avg_rate": int(round(base_rate * len(g)))}
        for m in WALK_MULTIPLES:
            q = CE / (CE + m * CE)
            rec[f"ind_{m}"] = int(np.quantile(ind, q, method="inverted_cdf"))
            rec[f"blk_{m}"] = int(np.quantile(blk, q, method="inverted_cdf"))
        rows.append(rec)
    res = pd.DataFrame(rows)

    print("\n" + "=" * 64)
    print("NIGHTLY OVERBOOKING BACKTEST (test period)")
    print("=" * 64)
    print(f"Hotel nights scored: {len(res):,}  "
          f"({res['night'].min().date()} to {res['night'].max().date()}), "
          f"{int(res['booked'].sum()):,} booked room-nights")

    # --- Does the night-level forecast hold up? -------------------------------
    def diagnose(sub, label):
        mae = (sub["expected"] - sub["actual"]).abs().mean()
        print(f"\n{label}: {len(sub)} nights. Average night: {sub['booked'].mean():.0f} bookings, "
              f"{sub['actual'].mean():.1f} actually cancelled, model expected {sub['expected'].mean():.1f}; "
              f"mean error {mae:.1f} rooms")
        for name, lo, hi, sd in [("independent", "ind_lo", "ind_hi", "ind_sd"),
                                 ("block-aware", "blk_lo", "blk_hi", "blk_sd")]:
            inside = ((sub["actual"] >= sub[lo]) & (sub["actual"] <= sub[hi])).mean()
            print(f"  {name:<12} simulated spread {sub[sd].mean():5.1f} rooms (st. dev.); "
                  f"actual inside the 10-90% band on {inside:.0%} of nights (target 80%)")
    diagnose(res, "Both hotels")
    for hotel, sub in res.groupby("hotel"):
        diagnose(sub, hotel)

    # --- Cost of each policy -----------------------------------------------
    def evaluate(k, x, m):
        walked = np.maximum(k - x, 0)
        empty = np.maximum(x - k, 0)
        return walked, empty, m * CE * walked + CE * empty

    for m in WALK_MULTIPLES:
        out = {}
        policies = {f"model, independent": res[f"ind_{m}"], f"model, block-aware": res[f"blk_{m}"],
                    "fixed 10% rule": res["fixed10"], "average-rate rule": res["avg_rate"],
                    "oracle (knows truth)": res["actual"]}
        for name, k in policies.items():
            w, e, c = evaluate(k.to_numpy(), res["actual"].to_numpy(), m)
            out[name] = {"walked/night": w.mean(), "empty/night": e.mean(),
                         "nights with a walk": (w > 0).mean(), "cost/night": c.mean()}
        print(f"\nWalk costs {m}x an empty room (cost in empty-room units):")
        print(pd.DataFrame(out).T.round(2).to_string())
        if m == PRIMARY_MULTIPLE:
            primary = pd.DataFrame(out).T

    # --- Same comparison, hotel by hotel (primary scenario) -------------------
    m = PRIMARY_MULTIPLE
    for hotel, sub in res.groupby("hotel"):
        out = {}
        policies = {"model, independent": sub[f"ind_{m}"], "model, block-aware": sub[f"blk_{m}"],
                    "fixed 10% rule": sub["fixed10"], "average-rate rule": sub["avg_rate"]}
        for name, k in policies.items():
            w, e, c = evaluate(k.to_numpy(), sub["actual"].to_numpy(), m)
            out[name] = {"walked/night": w.mean(), "empty/night": e.mean(),
                         "nights with a walk": (w > 0).mean(), "cost/night": c.mean()}
        print(f"\n{hotel}, walk costs {m}x an empty room:")
        print(pd.DataFrame(out).T.round(2).to_string())

    # --- Figure for the slide ------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(primary.index[::-1], primary["cost/night"][::-1], color="#3b6ea5")
    ax.set_xlabel(f"Average cost per hotel night (walk = {PRIMARY_MULTIPLE}x empty room)")
    ax.set_title("Overbooking rules on the test period")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "overbooking_cost.png", dpi=150)
    plt.close(fig)
    return res


# ---------------------------------------------------------------------------
# 5. Error analysis
# ---------------------------------------------------------------------------
def error_analysis(df, labels, prob, threshold):
    """Where does the model go wrong on test bookings, and why?

    False alarm = predicted cancel, guest stayed (the costly error: the hotel
    would overbook and walk the guest). Miss = predicted stay, guest cancelled
    (cheap: an empty room).
    """
    te = labels == "test"
    d = df[te].copy()
    d["p"] = prob[te]
    d["flag"] = d["p"] >= threshold
    d["fa"] = d["flag"] & (d[config.TARGET] == 0)
    d["miss"] = ~d["flag"] & (d[config.TARGET] == 1)
    # Duplicate status is judged against the whole dataset, not just the test rows.
    d["is_copy"] = df.duplicated(subset=original_columns(df), keep=False)[te]
    d["lead band"] = pd.cut(d["lead_time"], [-1, 7, 30, 90, 180, 800],
                            labels=["0-7d", "8-30d", "31-90d", "91-180d", "181d+"])

    print("\n" + "=" * 64)
    print(f"ERROR ANALYSIS at threshold {threshold:.2f} (test set, {len(d):,} bookings)")
    print("=" * 64)
    tp = int((d["flag"] & (d[config.TARGET] == 1)).sum())
    fa, ms = int(d["fa"].sum()), int(d["miss"].sum())
    tn = len(d) - tp - fa - ms
    print(f"Caught cancellations (TP) {tp:,} | false alarms (FP) {fa:,} | "
          f"missed cancellations (FN) {ms:,} | correct stays (TN) {tn:,}")
    print(f"Precision {tp / max(tp + fa, 1):.1%}   recall {tp / max(tp + ms, 1):.1%}")

    def slice_table(col):
        g = d.groupby(col, observed=True)
        t = pd.DataFrame({"bookings": g.size(), "cancel rate": g[config.TARGET].mean(),
                          "flagged": g["flag"].sum(), "false alarms": g["fa"].sum(),
                          "missed": g["miss"].sum()})
        t["precision"] = (t["flagged"] - t["false alarms"]) / t["flagged"].replace(0, np.nan)
        t["recall"] = (t["flagged"] - t["false alarms"]) / (
            (t["flagged"] - t["false alarms"]) + t["missed"])
        return t.round(3)

    for col in ["hotel", "market_segment", "deposit_type", "lead band", "is_copy"]:
        print(f"\nBy {col}:")
        print(slice_table(col).to_string())

    show = ["hotel", "market_segment", "deposit_type", "agent", "country", "lead_time",
            "adr", "total_of_special_requests", "previous_cancellations", "p", config.TARGET]
    print("\nMost confident FALSE ALARMS (flagged to cancel, guest stayed):")
    print(d[d["fa"]].sort_values("p", ascending=False)[show].head(8).round(2).to_string())
    print("\nMost confident MISSES (predicted to stay, guest cancelled):")
    print(d[d["miss"]].sort_values("p")[show].head(8).round(2).to_string())
    return d


def main():
    np.random.seed(config.SEED)
    df, labels = get_split()
    prob, raw, source = get_probabilities(df, labels)
    print(f"Probabilities from: {source}\n")
    calibration_report(df, labels, prob, raw)
    threshold = threshold_report(df, labels, prob)
    overbooking_backtest(df, labels, prob)
    error_analysis(df, labels, prob, threshold)


if __name__ == "__main__":
    main()
