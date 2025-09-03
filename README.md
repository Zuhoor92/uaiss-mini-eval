# UAISS-mini-eval

Evaluation script and synthetic dataset for the **Unified Adaptive IoD Security Stack (UAISS)**.

## Contents
- `uaiss_eval.py` – Python script to generate synthetic UAV telemetry, inject attack scenarios (GPS spoofing, command injection, telemetry anomaly), and run UAISS detection layers.
- `uav_timeseries.csv` – Generated dataset (600s UAV flight with injected attacks).
- `results_summary.csv` – Precision, Recall, F1 metrics for each attack type.
- Figures (`.png`) – Visual results of the evaluation, including detection metrics, timelines, and layer coverage.

## Usage
1. Run the evaluation script:
   ```bash
   python3 uaiss_eval.py
2. Results will be saved as:
   - Figures in `assets/`
   - `results_summary.csv` with metrics
   - `uav_timeseries.csv` with synthetic UAV flight data  

## Environment
- Python 3.8+  
- NumPy  
- Matplotlib  

Install dependencies if needed:
```bash
pip install numpy matplotlib
## License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
