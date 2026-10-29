"""
Browser tests: open the demo on localhost, visit all four pages, use the
main controls, and compare what is shown with the teammates' own code.
"""

import json
import re

import pytest

from conftest import PAGES, ROOT, assert_no_errors, metric, open_page, wait_until_idle


@pytest.fixture(scope="module")
def team():
    """The real data, split and model, loaded the same way app.py loads them."""
    import arfi_overbooking as ob
    import config
    import predict
    from split import get_split
    df, labels = get_split()
    return {"df": df, "labels": labels, "bundle": predict.load_model(),
            "predict": predict, "ob": ob, "config": config}


def move_walk_slider(page, steps):
    """Move the sidebar 'cost of walking a guest' slider by `steps` options."""
    thumb = page.get_by_test_id("stSidebar").get_by_role("slider")
    thumb.focus()
    for _ in range(abs(steps)):
        thumb.press("ArrowRight" if steps > 0 else "ArrowLeft")
    page.wait_for_timeout(300)
    wait_until_idle(page)


def click_main_radio(page, label):
    page.get_by_test_id("stMain").get_by_text(label, exact=True).click()
    page.wait_for_timeout(300)
    wait_until_idle(page)


# ---------------------------------------------------------------------------
# Every page loads
# ---------------------------------------------------------------------------
def test_sidebar_lists_the_four_pages(app):
    options = app.get_by_test_id("stSidebar").get_by_test_id("stRadioOption")
    assert [t.strip() for t in options.all_inner_texts()] == PAGES


@pytest.mark.parametrize("name, heading", [
    ("Predict a booking", "Will this booking cancel?"),
    ("Overbooking for a night", "How many rooms can we overbook tonight?"),
    ("Model results", "Model results"),
    ("Data and checks", "The data and the checks behind the model"),
])
def test_page_loads_without_errors(app, name, heading):
    open_page(app, name)
    assert app.locator("h2").first.inner_text().strip() == heading
    assert_no_errors(app)


# ---------------------------------------------------------------------------
# Page 1: predict a booking
# ---------------------------------------------------------------------------
def test_predict_page_shows_the_full_form(app, team):
    open_page(app, "Predict a booking")
    b = team["bundle"]
    assert app.get_by_test_id("stSelectbox").count() == len(b["categorical"])
    assert app.get_by_test_id("stNumberInput").count() == len(b["numeric"])
    assert re.fullmatch(r"\d+%", metric(app, "Chance this booking cancels"))


def test_loaded_booking_matches_the_team_model(app, team):
    """Load a random test booking; the % shown must equal predict.py's answer."""
    open_page(app, "Predict a booking")
    app.get_by_role("button", name=re.compile("Load a real booking")).click()
    app.wait_for_timeout(300)
    wait_until_idle(app)
    assert_no_errors(app)

    caption = app.get_by_text(re.compile(r"Loaded booking #\d+")).inner_text()
    rid = int(re.search(r"#(\d+)", caption).group(1))
    df, labels, cfg = team["df"], team["labels"], team["config"]
    assert labels[rid] == "test", "example booking must come from the test set"
    truth = "cancelled" if df.loc[rid, cfg.TARGET] else "stayed"
    assert f"guest {truth}" in caption

    expected = float(team["predict"].predict_proba(df.loc[[rid]], team["bundle"])[0])
    assert metric(app, "Chance this booking cancels") == f"{expected:.0%}"


def test_walk_cost_slider_moves_the_threshold(app, team):
    ob = team["ob"]
    open_page(app, "Predict a booking")
    label = "Flag-as-cancelling threshold"
    assert metric(app, label) == f"{ob.cost_threshold(ob.PRIMARY_MULTIPLE):.2f}"
    i = ob.WALK_MULTIPLES.index(ob.PRIMARY_MULTIPLE)
    move_walk_slider(app, +1)
    try:
        assert metric(app, label) == f"{ob.cost_threshold(ob.WALK_MULTIPLES[i + 1]):.2f}"
        assert_no_errors(app)
    finally:
        move_walk_slider(app, -1)          # leave the shared tab as it was


# ---------------------------------------------------------------------------
# Page 2: overbooking for a night
# ---------------------------------------------------------------------------
def test_night_page_uses_the_team_model(app):
    open_page(app, "Overbooking for a night")
    assert app.get_by_text("team model (outputs/predictions.csv)", exact=False).count() >= 1
    stand_in = app.get_by_test_id("stAlertContentWarning").filter(has_text="stand-in")
    assert stand_in.count() == 0, "app fell back to the stand-in model"
    assert int(metric(app, "Bookings for the night")) > 0
    assert re.fullmatch(r"\d+ rooms", metric(app, "Suggested overbooking"))
    assert app.get_by_test_id("stDataFrame").count() == 1
    assert app.get_by_test_id("stVegaLiteChart").count() == 1


def test_night_page_independent_mode(app):
    open_page(app, "Overbooking for a night")
    click_main_radio(app, "Independent")
    try:
        assert_no_errors(app)
        assert re.fullmatch(r"\d+ rooms", metric(app, "Suggested overbooking"))
    finally:
        click_main_radio(app, "Block-aware")


def test_dearer_walk_never_overbooks_more(app, team):
    """A walked guest costs more, so the suggestion must not go up."""
    ob = team["ob"]
    open_page(app, "Overbooking for a night")

    def rooms():
        return int(metric(app, "Suggested overbooking").split()[0])

    base = rooms()
    steps = len(ob.WALK_MULTIPLES) - 1 - ob.WALK_MULTIPLES.index(ob.PRIMARY_MULTIPLE)
    move_walk_slider(app, +steps)          # to the dearest walk cost
    try:
        assert rooms() <= base
        assert_no_errors(app)
    finally:
        move_walk_slider(app, -steps)


# ---------------------------------------------------------------------------
# Page 3: model results
# ---------------------------------------------------------------------------
def test_results_page_shows_saved_results(app):
    open_page(app, "Model results")
    for name in ["bhooshan_results.json", "bhooshan_results_no_country.json"]:
        saved = json.loads((ROOT / "outputs" / name).read_text(encoding="utf-8"))
        shown = app.get_by_text(re.compile(rf"Selected on validation: {saved['selected_model']}"))
        assert shown.count() >= 1
    # the two score tables and the hotel/country table
    assert app.get_by_test_id("stDataFrame").count() == 3


def test_results_page_figures_actually_load(app):
    open_page(app, "Model results")
    images = app.get_by_test_id("stImage").locator("img")
    assert images.count() == 3
    for i in range(3):
        img = images.nth(i)
        img.scroll_into_view_if_needed()
        app.wait_for_function("img => img.complete && img.naturalWidth > 0",
                              arg=img.element_handle(), timeout=30_000)


# ---------------------------------------------------------------------------
# Page 4: data and checks
# ---------------------------------------------------------------------------
def test_data_page_numbers_match_the_data(app, team):
    open_page(app, "Data and checks")
    df, cfg = team["df"], team["config"]
    assert metric(app, "Bookings") == f"{len(df):,}"
    assert metric(app, "Cancelled") == f"{df[cfg.TARGET].mean():.1%}"
    assert app.get_by_test_id("stDataFrame").count() == 2
    assert app.get_by_test_id("stVegaLiteChart").count() == 2


# ---------------------------------------------------------------------------
# Last: the server printed no Python errors during all of the above
# ---------------------------------------------------------------------------
def test_server_log_has_no_tracebacks(server, app):
    if server["log"] is None:
        pytest.skip("testing an external DEMO_URL; its log is not visible here")
    log = server["log"].read_text(encoding="utf-8", errors="replace")
    assert "Traceback" not in log, log
    assert "use_container_width" not in log, "deprecated Streamlit argument used"
