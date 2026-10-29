# Team J - Hotel cancellation prediction for safer overbooking

MA6700 Applied AI and Machine Learning, Assignment 3.

We predict which hotel bookings will be cancelled and turn those predictions
into a nightly overbooking number. The Streamlit demo (`app.py`) shows the
team's real model, data and results; nothing on screen is typed in by hand.

## Setup
1. Python 3.11 or newer.
2. `pip install -r requirements.txt streamlit`
3. The dataset is at `data/hotels_dataset.csv`
   (source: Antonio, de Almeida & Nunes (2019), Data in Brief 22, 41-49,
   doi:10.1016/j.dib.2018.11.126, CC BY 4.0; downloaded from
   https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-02-11).

## Run the analysis
`python main.py` runs every step in order and writes all results to `outputs/`.
Seed 42 is set in `config.py`. Each step can also be run on its own:

| Step | Command | What it does |
|---|---|---|
| 1 | `python data.py` | Data quality report |
| 2 | `python split.py --lookahead` | Time split (saved once in outputs/split.csv) and the look-ahead check |
| 3 | `python preprocessing_examples.py` | Preprocessing examples |
| 4 | `python leakage.py` | Leaking columns, baseline before and after |
| 5 | `python arfi_duplicates.py` | Duplicate rows: evidence and baseline both ways |
| 6 | `python bhooshan_models.py` | The three models, final test evaluation, saved model and predictions |
| 7 | `python subgroups.py` | Results by hotel and country (ethics) |
| 8 | `python arfi_overbooking.py` | Calibration, nightly overbooking backtest, error analysis |
| 9 | `python predict.py` | Final model check on three test bookings |

## Run the demo
After `python main.py` has filled `outputs/`:

```
python -m streamlit run app.py
```

The demo has four pages: predict one booking, the overbooking number for one
hotel night, the model results, and the data checks.

## Test the demo
The tests start the demo on a localhost port and click through all four pages
in headless Chrome, checking the numbers shown against the team's own code.

```
pip install -r tests/requirements-test.txt
python -m pytest tests -v
```

To test a demo that is already running, set `DEMO_URL`, for example
`DEMO_URL=http://localhost:8501`. Set `DEMO_HEADED=1` to watch the browser.

## Deploy on Streamlit Community Cloud
Use `deploy/streamlit_app.py` as the main file. It runs `app.py`, and Streamlit
Cloud installs the slim `deploy/requirements.txt` next to it. That file leaves
out PyTorch, which only the training step needs. The saved model and results
in `outputs/` are part of the repository, so the cloud app does not retrain.
