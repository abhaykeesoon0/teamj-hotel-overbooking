"""
deploy/streamlit_app.py - entry point for Streamlit Community Cloud.

Streamlit Cloud installs the requirements file that sits next to the entry
point, so this folder carries a slim deploy/requirements.txt with only what
the demo needs. The team's requirements.txt at the root also lists PyTorch,
which is used for training (main.py) but never by the demo, and would make
the cloud build several GB larger.

This file just runs the real demo, app.py, from the project root.
Locally, keep using:   streamlit run app.py

Owner: Abhay (demo UI).
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))      # so app.py can import config, predict, ...

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
