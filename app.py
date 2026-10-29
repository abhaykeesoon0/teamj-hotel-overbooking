"""
app.py - Team J demo: hotel booking cancellations -> nightly overbooking.

Put this file in the project folder (next to config.py, predict.py,
arfi_overbooking.py ...) and run:   streamlit run app.py
Run `python main.py` once first, so outputs/ holds the saved model, the
predictions and the result files. Everything on screen comes from the
teammates' own code and files; nothing is typed in by hand.

Owner: Abhay (demo UI).
"""

import json
import random

import numpy as np
import pandas as pd
import streamlit as st

import arfi_overbooking as ob
import config
import predict
from arfi_duplicates import original_columns
from split import get_split

OUT = config.OUTPUT_DIR
TARGET = config.TARGET

st.set_page_config(page_title="Team J - Hotel Overbooking", page_icon="🏨", layout="wide")


# ---------------------------------------------------------------------------
# Loaders (cached, so the app stays fast after the first load)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading the dataset and the saved split...")
def load_data():
    return get_split()                      # (bookings, set label per booking)


@st.cache_resource(show_spinner="Loading the saved model...")
def load_bundle():
    return predict.load_model()


@st.cache_data(show_spinner="Scoring validation and test bookings...")
def load_probabilities():
    df, labels = load_data()
    prob, _raw, source = ob.get_probabilities(df, labels)   # Arfi's function
    return prob, source


@st.cache_data(show_spinner="Building the hotel nights...")
def load_nights():
    df, labels = load_data()
    prob, _ = load_probabilities()
    return ob.build_nights(df, labels, prob)                # Arfi's function


@st.cache_data
def copy_flags():
    df, _ = load_data()
    return df.duplicated(subset=original_columns(df), keep="first")


def read_json(name):
    path = OUT / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


df, labels = load_data()
test = df[labels == "test"]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🏨 Team J")
st.sidebar.caption("Predict cancellations, then decide how many rooms to overbook.")
page = st.sidebar.radio("Page", ["Predict a booking", "Overbooking for a night",
                                 "Model results", "Data and checks"])
walk = st.sidebar.select_slider(
    "Cost of walking a guest (in empty rooms)", options=ob.WALK_MULTIPLES,
    value=ob.PRIMARY_MULTIPLE,
    help="If a guest is turned away it costs this many times an empty room. "
         "A dearer walk makes the model more cautious.")


# ---------------------------------------------------------------------------
# Page 1: one booking -> cancellation risk
# ---------------------------------------------------------------------------
def is_int(col):
    return pd.api.types.is_integer_dtype(df[col])


def cat_options(col):
    """Values seen in the data, most frequent first; missing shows as (none)."""
    counts = df[col].astype("string").value_counts(dropna=False)
    return ["(none)" if pd.isna(v) else str(v) for v in counts.index]


def load_example(num, cat):
    """Callback: fill the form with a real booking from the test set."""
    rid = random.choice(list(test.index))
    row = test.loc[rid]
    for c in num:
        v = 0 if pd.isna(row[c]) else row[c]
        st.session_state[f"in_{c}"] = int(v) if is_int(c) else float(v)
    for c in cat:
        st.session_state[f"in_{c}"] = "(none)" if pd.isna(row[c]) else str(row[c])
    st.session_state["example"] = (int(rid), int(row[TARGET]))


def page_predict():
    st.header("Will this booking cancel?")
    try:
        bundle = load_bundle()
    except FileNotFoundError as err:
        st.error(f"{err}\n\nThe file is only written when the selected model is "
                 "logistic regression or CatBoost.")
        return
    num, cat = bundle["numeric"], bundle["categorical"]

    for c in num:                                   # first-run defaults
        if f"in_{c}" not in st.session_state:
            m = df[c].median()
            st.session_state[f"in_{c}"] = int(m) if is_int(c) else float(m)
    for c in cat:
        if f"in_{c}" not in st.session_state:
            st.session_state[f"in_{c}"] = cat_options(c)[0]

    st.button("🎲 Load a real booking from the test set", on_click=load_example,
              args=(num, cat))
    if "example" in st.session_state:
        rid, actual = st.session_state["example"]
        st.caption(f"Loaded booking #{rid}. In the real data this guest "
                   f"**{'cancelled' if actual else 'stayed'}**. Edit any field "
                   "and the answer updates.")

    cols = st.columns(3)
    for i, c in enumerate(cat):
        with cols[i % 3]:
            st.selectbox(c.replace("_", " "), cat_options(c), key=f"in_{c}")
    cols = st.columns(4)
    for i, c in enumerate(num):
        with cols[i % 4]:
            lo, hi = df[c].min(), df[c].max()
            if is_int(c):
                st.number_input(c.replace("_", " "), int(lo), int(hi), step=1, key=f"in_{c}")
            else:
                st.number_input(c.replace("_", " "), float(lo), float(hi), step=1.0, key=f"in_{c}")

    row = {c: st.session_state[f"in_{c}"] for c in num}
    row.update({c: None if st.session_state[f"in_{c}"] == "(none)"
                else st.session_state[f"in_{c}"] for c in cat})
    p = float(predict.predict_proba(pd.DataFrame([row]), bundle)[0])
    thr = ob.cost_threshold(walk)

    st.divider()
    a, b, c3 = st.columns(3)
    a.metric("Chance this booking cancels", f"{p:.0%}")
    b.metric("Average in the test set", f"{test[TARGET].mean():.0%}")
    c3.metric("Flag-as-cancelling threshold", f"{thr:.2f}",
              help=f"Walk = {walk}x an empty room, so we only count on a "
                   f"cancellation above {thr:.0%}.")
    st.progress(min(max(p, 0.0), 1.0))
    if p > thr:
        st.warning("Counted on to cancel: the hotel could overbook against this booking.")
    else:
        st.success("Not counted on to cancel: no overbooking against this booking.")
    st.caption(f"Model: {bundle['name']} (calibrated on validation).")

    if bundle["name"] == "catboost":
        with st.expander("What the model pays attention to (all bookings, not just this one)"):
            m = bundle["fitted"]["model"]
            imp = pd.Series(m.get_feature_importance(), index=m.feature_names_)
            st.bar_chart(imp.sort_values(ascending=False).head(10))


# ---------------------------------------------------------------------------
# Page 2: one hotel night -> overbooking number
# ---------------------------------------------------------------------------
def page_night():
    st.header("How many rooms can we overbook tonight?")
    nights = load_nights()
    _, source = load_probabilities()
    st.caption(f"Probabilities from: {source}")
    if source.startswith("STAND-IN"):
        st.warning("outputs/predictions.csv was not found, so a stand-in model is "
                   "used. Run bhooshan_models.py to use the team model.")

    first, last = nights["night"].min().date(), nights["night"].max().date()
    c1, c2, c3 = st.columns(3)
    hotel = c1.selectbox("Hotel", sorted(nights["hotel"].unique()))
    day = c2.date_input("Night", value=first + (last - first) / 2,
                        min_value=first, max_value=last)
    mode = c3.radio("Cancellations are simulated as",
                    ["Block-aware", "Independent"], horizontal=True,
                    help="Block-aware: identical bookings are one reservation "
                         "covering several rooms, so they cancel together.")

    stamp = pd.Timestamp(day)
    g = nights[(nights["hotel"] == hotel) & (nights["night"] == stamp)]
    if g.empty:
        st.info("No bookings on that night.")
        return

    # Same random stream as the backtest in arfi_overbooking.py
    rng = np.random.default_rng([config.SEED, int(stamp.value // 86_400_000_000_000),
                                 int(hotel == "City Hotel")])
    ind, blk = ob.simulate_cancellations(g["prob"].to_numpy(), g["unit"].to_numpy(), rng)
    sims = blk if mode == "Block-aware" else ind
    k = int(np.quantile(sims, ob.CE / (ob.CE + walk * ob.CE), method="inverted_cdf"))
    lo, hi = int(np.quantile(sims, 0.10)), int(np.quantile(sims, 0.90))

    actual = int(g[TARGET].sum())
    n = len(g)
    base_rate = df.loc[labels == "train", TARGET].mean()

    a, b, c4, d = st.columns(4)
    a.metric("Bookings for the night", n)
    b.metric("Model expects to cancel", f"{g['prob'].sum():.0f}",
             help=f"Likely range {lo}-{hi} (10-90%).")
    c4.metric("Suggested overbooking", f"{k} rooms")
    d.metric("Actually cancelled (real data)", actual)

    st.subheader("What each rule would have done on this night")
    rules = {"Team model": k,
             "Fixed 10% rule": int(ob.FIXED_RULE_SHARE * n),
             "Average-rate rule": int(round(base_rate * n)),
             "Oracle (knows the truth)": actual}
    table = pd.DataFrame({
        "Rooms overbooked": rules,
        "Guests walked": {r: max(v - actual, 0) for r, v in rules.items()},
        "Rooms left empty": {r: max(actual - v, 0) for r, v in rules.items()},
    })
    table["Cost (empty rooms)"] = walk * table["Guests walked"] + table["Rooms left empty"]
    st.dataframe(table, width="stretch")

    st.subheader("Simulated number of cancellations")
    st.bar_chart(pd.Series(sims).value_counts().sort_index())
    st.caption(f"Walk = {walk}x an empty room: overbook by the number of cancellations "
               f"we are about {ob.CE / (ob.CE + walk * ob.CE):.0%} sure to reach. "
               "The backtest over every test-period night is on the Model results page.")


# ---------------------------------------------------------------------------
# Page 3: the team's saved results
# ---------------------------------------------------------------------------
def score_table(r):
    def col(key, metric):
        return pd.DataFrame(r[key]).T[metric]
    return pd.DataFrame({
        "Validation PR-AUC": col("validation_scores", "average_precision"),
        "Test PR-AUC": col("test_scores", "average_precision"),
        "Test PR-AUC (no copies)": col("test_scores_deduplicated", "average_precision"),
        "Test Brier (lower is better)": col("test_scores", "brier"),
    }).round(3)


def page_results():
    st.header("Model results")
    for title, name in [("With country", "bhooshan_results.json"),
                        ("Without country (ethics comparison)", "bhooshan_results_no_country.json")]:
        r = read_json(name)
        st.subheader(title)
        if r is None:
            st.info(f"{name} not found in outputs/. Run bhooshan_models.py.")
            continue
        st.write(f"Selected on validation: **{r['selected_model']}**")
        st.dataframe(score_table(r), width="stretch")

    st.subheader("Calibration, cost and duplicates")
    cols = st.columns(3)
    for col, (name, cap) in zip(cols, [
            ("reliability.png", "Predicted vs actual cancel rate (test set)"),
            ("overbooking_cost.png", "Cost per night of each overbooking rule"),
            ("duplicates_by_segment.png", "Duplicate rows by market segment")]):
        if (OUT / name).exists():
            col.image(str(OUT / name), caption=cap)
        else:
            col.info(f"{name} not found.")

    sub = OUT / "subgroups.csv"
    if sub.exists():
        st.subheader("Results by hotel and country")
        st.dataframe(pd.read_csv(sub), width="stretch")


# ---------------------------------------------------------------------------
# Page 4: the data and the checks behind the model
# ---------------------------------------------------------------------------
def page_data():
    st.header("The data and the checks behind the model")
    copies = copy_flags()
    a, b, c, d = st.columns(4)
    a.metric("Bookings", f"{len(df):,}")
    b.metric("Cancelled", f"{df[TARGET].mean():.1%}")
    c.metric("City Hotel cancelled", f"{df.loc[df.hotel == 'City Hotel', TARGET].mean():.1%}")
    d.metric("Rows that are copies", f"{copies.mean():.1%}")

    st.subheader("Time split (train, validation, test)")
    split = df.groupby(labels).agg(bookings=(TARGET, "size"), first_arrival=("arrival_date", "min"),
                                   last_arrival=("arrival_date", "max"),
                                   cancel_rate=(TARGET, "mean"))
    for col in ["first_arrival", "last_arrival"]:          # dates only, no 00:00:00
        split[col] = split[col].dt.strftime("%Y-%m-%d")
    st.dataframe(split.reindex(["train", "val", "test"]).round({"cancel_rate": 3}), width="stretch")

    st.subheader("Cancellation rate by arrival month")
    month = df.groupby(df["arrival_date"].dt.to_period("M").astype(str))[TARGET].mean()
    st.line_chart(month)

    st.subheader("Columns removed because they leak the outcome")
    st.write(", ".join(f"`{c}`" for c in config.LEAKING_COLUMNS + config.PRECAUTIONARY_DROPS))
    changed = df["reserved_room_type"] != df["assigned_room_type"]
    zero = df["booking_changes"] == 0
    parking = df["required_car_parking_spaces"] > 0
    ev = pd.DataFrame([
        ("Room changed at check-in", changed),
        ("Room not changed", ~changed),
        ("Parking requested", parking),
        ("0 booking changes", zero),
        ("1+ booking changes", ~zero)], columns=["Group", "mask"])
    ev["Bookings"] = [int(m.sum()) for m in ev["mask"]]
    ev["Cancel rate"] = [df.loc[m, TARGET].mean() for m in ev["mask"]]
    st.dataframe(ev.drop(columns="mask").round(3), width="stretch")
    st.caption("These values are only known once the outcome is known, so a model "
               "using them would be reading the answer.")

    st.subheader("Where the duplicate rows sit")
    st.bar_chart(copies.groupby(df["market_segment"]).mean().sort_values())


{"Predict a booking": page_predict, "Overbooking for a night": page_night,
 "Model results": page_results, "Data and checks": page_data}[page]()
