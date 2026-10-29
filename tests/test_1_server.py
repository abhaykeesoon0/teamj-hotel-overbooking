"""The Streamlit server itself answers on localhost."""

import urllib.request


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read().decode("utf-8", "replace")


def test_health_endpoint_says_ok(server):
    status, body = get(f"{server['url']}/_stcore/health")
    assert status == 200 and body.strip() == "ok"


def test_front_page_is_served(server):
    status, body = get(server["url"] + "/")
    assert status == 200
    assert "<html" in body.lower()
