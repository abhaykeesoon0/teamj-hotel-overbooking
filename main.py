"""
main.py - the single entry point: runs the whole Team J pipeline in order.

    python main.py

Each step is its own script, so it can also be run and read on its own.
They run in this order because each one uses what the previous ones made:
the split must exist before any model, the models must save predictions
before the overbooking analysis can use them, and so on. If any step fails,
the run stops there and says which one, rather than continuing on missing
results.

All numbers are printed to the screen and every figure and table is saved
in outputs/. Seed 42 (config.py) makes the run repeatable.

Owner: Nohur Muhammad Jamiil (team leader).
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Data quality report", ["data.py"]),
    ("Time split + look-ahead check (condition 2)", ["split.py", "--lookahead"]),
    ("Preprocessing examples", ["preprocessing_examples.py"]),
    ("Leaking columns (condition 1)", ["leakage.py"]),
    ("Duplicate rows (condition 3)", ["arfi_duplicates.py"]),
    ("Models, selection on validation, final test", ["bhooshan_models.py"]),
    ("Results by hotel and country (ethics)", ["subgroups.py"]),
    ("Calibration, overbooking backtest, error analysis", ["arfi_overbooking.py"]),
    ("Final model check on three test bookings", ["predict.py"]),
]


def main():
    start = time.time()
    for number, (title, command) in enumerate(STEPS, start=1):
        print("\n" + "#" * 78)
        print(f"# STEP {number}/{len(STEPS)}: {title}  ({' '.join(command)})")
        print("#" * 78, flush=True)
        result = subprocess.run([sys.executable, *command], cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"\nStopped: step {number} ({command[0]}) failed. "
                     "Fix it before running the later steps.")
    print(f"\nAll {len(STEPS)} steps finished in {time.time() - start:.0f} s. "
          "Results are in the outputs/ folder.")


if __name__ == "__main__":
    main()
