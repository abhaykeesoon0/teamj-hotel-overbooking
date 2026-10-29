"""
Shared setup for the demo tests: start app.py on a localhost port and open
it in a real (headless) browser.

Run from the project folder:
    python -m pytest tests -v

Options (environment variables):
    DEMO_URL=http://localhost:8501   test an app that is already running
                                     instead of starting a new one
    DEMO_BROWSER=msedge              use Edge instead of Chrome
    DEMO_HEADED=1                    show the browser window while testing

Owner: Abhay (demo UI).
"""

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))           # so tests can import the team's modules

PAGES = ["Predict a booking", "Overbooking for a night",
         "Model results", "Data and checks"]
LOAD_TIMEOUT_MS = 180_000               # first run loads 119k bookings


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _healthy(url):
    try:
        with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as r:
            return r.read().decode().strip() == "ok"
    except OSError:
        return False


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    """URL of the running demo, plus the server log path (None if external)."""
    external = os.environ.get("DEMO_URL")
    if external:
        url = external.rstrip("/")
        if not _healthy(url):
            pytest.exit(f"DEMO_URL {url} is not answering /_stcore/health")
        yield {"url": url, "log": None}
        return

    for name in ["final_model.joblib", "predictions.csv", "split.csv",
                 "bhooshan_results.json", "bhooshan_results_no_country.json"]:
        if not (ROOT / "outputs" / name).exists():
            pytest.exit(f"outputs/{name} is missing. Run `python main.py` first.")

    port = _free_port()
    url = f"http://localhost:{port}"
    log = tmp_path_factory.mktemp("server") / "streamlit.log"
    with open(log, "w", encoding="utf-8") as out:
        proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py",
             "--server.headless", "true", "--server.port", str(port),
             "--browser.gatherUsageStats", "false"],
            cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 60
        while not _healthy(url):
            if proc.poll() is not None or time.time() > deadline:
                pytest.exit("Streamlit did not start:\n" + log.read_text(encoding="utf-8"))
            time.sleep(0.5)
        yield {"url": url, "log": log}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(channel=os.environ.get("DEMO_BROWSER", "chrome"),
                              headless=not os.environ.get("DEMO_HEADED"))
        yield b
        b.close()


@pytest.fixture(scope="module")
def app(server, browser):
    """One browser tab on the demo, shared by the tests in a module.

    page_errors collects JavaScript errors raised in the page, so a test can
    check the front end did not crash as well as the Python side.
    """
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.set_default_timeout(LOAD_TIMEOUT_MS)
    page.page_errors = []
    page.on("pageerror", lambda err: page.page_errors.append(str(err)))
    page.goto(server["url"])
    page.get_by_test_id("stSidebar").wait_for()
    wait_until_idle(page)
    yield page
    page.close()


# ---------------------------------------------------------------------------
# Helpers used by the tests
# ---------------------------------------------------------------------------
def wait_until_idle(page, settle_ms=800):
    """Wait until Streamlit has finished rerunning the script.

    While a rerun is in progress the old page's elements stay on screen
    (greyed out), so checks must wait for the 'running' icon to go away
    and stay away briefly.
    """
    running = page.get_by_test_id("stStatusWidgetRunningIcon")
    while True:
        running.wait_for(state="detached")
        page.wait_for_timeout(settle_ms)
        if running.count() == 0:
            return


def open_page(page, name):
    page.get_by_test_id("stSidebar").get_by_text(name, exact=True).click()
    page.wait_for_timeout(300)
    wait_until_idle(page)


def metric(page, label):
    """Displayed value of the st.metric whose label is `label`."""
    box = page.get_by_test_id("stMetric").filter(
        has=page.get_by_test_id("stMetricLabel").get_by_text(label, exact=True))
    return box.get_by_test_id("stMetricValue").inner_text().strip()


def assert_no_errors(page):
    """No Python exception box, no st.error box, no JavaScript error."""
    exc = page.get_by_test_id("stException")
    assert exc.count() == 0, "App raised an exception:\n" + "\n".join(exc.all_inner_texts())
    err = page.get_by_test_id("stAlertContentError")
    assert err.count() == 0, "App shows an error:\n" + "\n".join(err.all_inner_texts())
    assert not page.page_errors, f"JavaScript errors in the page: {page.page_errors}"
