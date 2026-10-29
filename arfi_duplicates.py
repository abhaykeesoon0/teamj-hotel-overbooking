"""
arfi_duplicates.py - Condition 3: investigate the exact duplicate rows.

Owner: Mohammad Arfi Ilshad Dolip.

What it does:
  1. Counts the duplicate rows and describes them: block sizes, the largest
     blocks, where they sit (hotel, segment, deposit, customer type, agent),
     how they differ from other bookings, and whether identical rows share
     an outcome. Saves outputs/duplicates_by_segment.png.
  2. Trains the baseline with and without the duplicates and scores both on
     the full and the de-duplicated VALIDATION set.

Integration changes (team leader) are marked "INTEGRATION".

Run:  python arfi_duplicates.py
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")   
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from split import get_split

TARGET_PRECISION = 0.90


def original_columns(df):
    """The 32 columns of the published file (excludes our two added dates).

    Duplicates are judged on these only, so the dates we rebuilt cannot make
    two otherwise identical rows look different.
    """
    return [c for c in df.columns if c not in ("arrival_date", "booking_date")]


# ---------------------------------------------------------------------------
# Steps 1 and 2: what the duplicates look like
# ---------------------------------------------------------------------------
def investigate(df):
    """Print how many copies there are and where they sit. Returns key numbers."""
    cols = original_columns(df)
    n = len(df)
    is_copy = df.duplicated(subset=cols, keep="first")   
    in_group = df.duplicated(subset=cols, keep=False)    
    out = {"rows": n, "copies": int(is_copy.sum())}

    print("=" * 64)
    print("CONDITION 3 - DUPLICATE ROWS")
    print("=" * 64)

    # --- Size and shape -----------------------------------------------------
    sizes = df[in_group].groupby(cols, dropna=False).size()
    copies_per_group = sizes - 1          
    big = copies_per_group[sizes >= 5].sum()
    print(f"Copies after the first: {out['copies']:,} ({out['copies'] / n:.1%} of rows)")
    print(f"Rows involved in a repeated group: {int(in_group.sum()):,}")
    print(f"Distinct repeated groups: {len(sizes):,}   largest group: {int(sizes.max())} identical rows")
    print(f"Groups of exactly 2 rows: {int((sizes == 2).sum()):,} "
          f"-> only {copies_per_group[sizes == 2].sum() / out['copies']:.1%} of all copies")
    print(f"Groups of 5 or more rows: {int((sizes >= 5).sum()):,} "
          f"-> {big / out['copies']:.1%} of all copies")
    # Why this matters: a random double-entry glitch would produce mostly pairs.
    # Mostly big blocks looks like one reservation covering many rooms.
    out["copies_in_big_groups"] = float(big / out["copies"])

    print("\nFive largest repeated groups (one example row each):")
    top = sizes.sort_values(ascending=False).head(5)
    show = ["hotel", "agent", "market_segment", "deposit_type", "arrival_date",
            "lead_time", "adr", "is_canceled"]
    rows = []
    for key, size in top.items():   
        mask = np.ones(n, dtype=bool)
        for col, val in zip(cols, key):
            mask &= (df[col].isna() if pd.isna(val) else df[col] == val).to_numpy()
        ex = df.loc[mask, show].iloc[0].copy()
        ex["rows_in_group"] = int(size)
        rows.append(ex)
    print(pd.DataFrame(rows).to_string(index=False))

    # --- Where they sit -----------------------------------------------------
    print("\nShare of rows that are copies, by group (compare with the overall "
          f"{out['copies'] / n:.1%}):")
    for col in ["hotel", "market_segment", "deposit_type", "customer_type"]:
        rate = is_copy.groupby(df[col]).mean()
        size = df[col].value_counts()
        tab = pd.DataFrame({"rows": size, "copy_share": rate.map("{:.1%}".format)})
        print(f"\n  by {col}:")
        print(tab.sort_values("rows", ascending=False).to_string())

    agents = df["agent"].fillna(-1)
    ag = pd.DataFrame({"copies": is_copy.groupby(agents).sum(),
                       "rows": agents.value_counts()}).dropna()
    ag["copy_share"] = ag["copies"] / ag["rows"]
    top5 = ag.sort_values("copies", ascending=False).head(5)
    print("\n  top 5 agents by number of copies (agent -1 = no agent):")
    print(top5.assign(copy_share=top5["copy_share"].map("{:.1%}".format)).to_string())
    print(f"  these 5 agents hold {top5['copies'].sum() / out['copies']:.1%} of all copies")

    # --- How the copies differ from the other bookings -----------------------
    uniq = df[~is_copy]
    cop = df[is_copy]
    print("\nCopies against non-copies:")
    print(f"  cancellation rate   copies {cop[config.TARGET].mean():.1%}   "
          f"non-copies {uniq[config.TARGET].mean():.1%}   all rows {df[config.TARGET].mean():.1%}")
    print(f"  median lead time    copies {cop['lead_time'].median():.0f} days   "
          f"non-copies {uniq['lead_time'].median():.0f} days")
    print(f"  no special requests copies {(cop['total_of_special_requests'] == 0).mean():.1%}   "
          f"non-copies {(uniq['total_of_special_requests'] == 0).mean():.1%}")
    hot = pd.DataFrame({"all rows": df.groupby("hotel")[config.TARGET].mean(),
                        "duplicates removed": uniq.groupby("hotel")[config.TARGET].mean()})
    print("  cancellation rate by hotel, before and after removing copies:")
    print(hot.map("{:.1%}".format).to_string())
    out["cancel_rate_all"] = float(df[config.TARGET].mean())
    out["cancel_rate_dedup"] = float(uniq[config.TARGET].mean())

    # --- Do identical bookings share an outcome? ------------------------------
    
    outcome_cols = [config.TARGET, "reservation_status", "reservation_status_date"]
    feats = [c for c in cols if c not in outcome_cols]
    g = df.groupby(feats, dropna=False)[config.TARGET].agg(["size", "mean"])
    g = g[g["size"] >= 2]
    purity = (np.maximum(g["mean"], 1 - g["mean"]) * g["size"]).sum() / g["size"].sum()
    mixed = int(((g["mean"] > 0) & (g["mean"] < 1)).sum())
    print(f"\nGroups identical on every booking-time feature: {len(g):,} "
          f"({int(g['size'].sum()):,} rows)")
    print(f"  groups with mixed outcomes (some cancelled, some stayed): {mixed:,}")
    print(f"  rows sharing their group's majority outcome: {purity:.1%}")
    out["outcome_agreement"] = float(purity)

    # --- Figure for the report  -------------------------------------
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    seg = is_copy.groupby(df["market_segment"]).mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(seg.index, seg.values * 100, color="#3b6ea5")
    ax.axvline(out["copies"] / n * 100, color="#b03a2e", ls="--", label="all rows")
    ax.set_xlabel("Rows that are exact copies (%)")
    ax.set_title("Where the duplicate rows sit, by market segment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "duplicates_by_segment.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure saved to {config.OUTPUT_DIR / 'duplicates_by_segment.png'}")
    return out, is_copy


# ---------------------------------------------------------------------------
# Step 3: does the decision change the score?
# ---------------------------------------------------------------------------
def make_baseline():
    """Logistic regression on scaled numbers plus one-hot categories.

    One-hot is fine for a baseline even for country/agent/company: categories
    seen fewer than 50 times are pooled so the column count stays sane. (The
    neural network will use embeddings instead.)
    """
    cats = config.LOW_CARDINALITY_CATEGORICAL + config.HIGH_CARDINALITY_CATEGORICAL
    pre = ColumnTransformer([
        ("num", StandardScaler(), config.NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), cats),
    ])
    return Pipeline([("prep", pre),
                     ("lr", LogisticRegression(max_iter=1000, random_state=config.SEED))])


def prepare(df):
    """Select the model inputs and fill the known gaps the way features.py will.

    NULL agent/company means no agent / not a company booking, so it becomes
    its own category rather than a missing value. Reservation_status and its
    date are dropped first because they are the answer (see data.py).
    """
    X = df.drop(columns=config.LEAKING_COLUMNS, errors="ignore")
    X = X[config.NUMERIC + config.LOW_CARDINALITY_CATEGORICAL + config.HIGH_CARDINALITY_CATEGORICAL].copy()
    X["children"] = X["children"].fillna(0)
    for col in config.HIGH_CARDINALITY_CATEGORICAL:
        X[col] = X[col].astype("object").where(X[col].notna(), "none").astype(str)
    return X


def pick_threshold(y_val, p_val):
    """Lowest threshold whose validation precision reaches TARGET_PRECISION.

    Chosen on validation only; the test set is never used to tune anything.
    """
    prec, _, thr = precision_recall_curve(y_val, p_val)
    ok = np.where(prec[:-1] >= TARGET_PRECISION)[0]
    return float(thr[ok[0]]) if len(ok) else float(thr[-1])


def score(y, p, thr):
    """PR-AUC, ROC-AUC, Brier, and precision/recall at the chosen threshold."""
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    return {"PR-AUC": average_precision_score(y, p), "ROC-AUC": roc_auc_score(y, p),
            "Brier": brier_score_loss(y, p),
            "precision": tp / max(int(pred.sum()), 1), "recall": tp / int((y == 1).sum())}


def baseline_both_ways(df, labels, is_copy):
    """Train with and without the copies; score both on both validation sets.

    The split is Jamiil's saved one. Removing copies only removes rows from it;
    no booking changes sets and no new split is made. Scoring each model on
    the full validation set AND on the de-duplicated one separates two effects:
    what duplicates do to training, and what they do to the score itself.
    """
    variants = {"keep duplicates": pd.Series(True, index=df.index),
                "remove duplicates": ~is_copy}
    models = {}
    for name, keep in variants.items():
        tr = (labels == "train") & keep
        va = (labels == "val") & keep
        m = make_baseline().fit(prepare(df[tr]), df.loc[tr, config.TARGET])
        thr = pick_threshold(df.loc[va, config.TARGET].to_numpy(),
                             m.predict_proba(prepare(df[va]))[:, 1])
        models[name] = (m, thr, int(tr.sum()))

    # INTEGRATION: scored on validation, not test. Team rule: only the final
    # model run touches the test set, once.
    tests = {"full validation set": labels == "val",
             "de-duplicated validation set": (labels == "val") & ~is_copy}
    print("\n" + "=" * 64)
    print(f"BASELINE (logistic regression), threshold set on validation for "
          f"{TARGET_PRECISION:.0%} precision")
    print("=" * 64)
    for tname, tmask in tests.items():
        y = df.loc[tmask, config.TARGET].to_numpy()
        print(f"\nScored on the {tname}: {int(tmask.sum()):,} bookings, "
              f"cancel rate {y.mean():.1%}")
        rows = {}
        for mname, (m, thr, ntrain) in models.items():
            p = m.predict_proba(prepare(df[tmask]))[:, 1]
            rows[f"trained: {mname} ({ntrain:,} rows)"] = score(y, p, thr)
        print(pd.DataFrame(rows).T.round(3).to_string())
    return models


def main():
    np.random.seed(config.SEED)
    df, labels = get_split()   # Jamiil's saved time split, never remade here
    stats, is_copy = investigate(df)
    baseline_both_ways(df, labels, is_copy)


if __name__ == "__main__":
    main()
