# THGB Implementation Guide

1. Verify Git LFS files under `data/thgb/DBLP-TA`.
2. Install requirements.
3. Run `python -m pytest -q`.
4. Launch `python -m streamlit run app.py`.
5. Run `python run_research.py --data-dir data/thgb/DBLP-TA`.
6. Inspect `results/thgb/thgb_results.csv` and `thgb_metadata.json`.

Before paper submission, document the consecutive-ID page-layout assumption,
synthetic workload construction, and aggregate-edge scoring limitation.
