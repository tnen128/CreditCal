# Federated Learning for Credit Risk Assessment 🏦

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Privacy-preserving mortgage default prediction using federated learning with LSTM models and probability calibration.

Replication and extension of *Lee et al. (2023)* — showing that federated learning matches centralized performance with zero data sharing, and that calibration is critical before deploying these models in practice.

---

## 📄 Paper

**Federated Learning for Credit Risk Assessment in Mortgage Lending** — *Lee et al. (2023)*

📥 [Download PDF](https://scholarspace.manoa.hawaii.edu/server/api/core/bitstreams/4f3e6c97-3e8d-4d5e-995f-3420a3e93c1f/content)

---

## 🏗️ How It Works

![FL Architecture](results/arch%20diagrams/fl_architecture_diagram.png)

Each of the 14 banks trains a local LSTM model on their own loan data. Only the model weights — never the raw records — are sent to a central server, which averages them (FedAvg) and sends the updated model back. After 100 rounds, you get a shared model that performs as well as if everyone pooled their data.

The LSTM takes a 60-month loan sequence (95 features per step) and outputs a default probability. After training, we apply four calibration methods to make those probabilities reliable enough for real-world use.

---

## 🚀 Quick Start

**Step 1 — Install dependencies**

```bash
git clone https://github.com/yourusername/fl-credit-risk-assessment.git
cd fl-credit-risk-assessment
pip install -r requirements.txt
```

**Step 2 — Get the data**

Register for a free account at https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset and download the origination and monthly performance files for 2006–2009. Place them under `data/raw/` in project root dir. You'll also need a few macroeconomic files (LAUS, FRED, FMHPI)

Once the raw files are in place, run the preprocessing script:

```bash
python src/preprocess.py
```

This takes 30–60 min and writes the cleaned train/val/test splits to `data/preprocessed_strict/`.

**Step 3 — Adjust parameters (optional)**

Open `src/train.py` and change the constants at the top of the file if needed:

```python
EPOCHS = 10       # local epochs per round
FL_ROUNDS = 30    # federated communication rounds  
MAX_LOANS_PER_INSTITUTION = 1000   # set lower for a quick test
```

For a fast sanity check, drop `FL_ROUNDS` to 1 and `EPOCHS` to 1 — runs the full pipeline in ~2 minutes.

**Step 4 — Run**

```bash
python src/train.py
```

Runs all five scenarios (Local, Central, FL(n), FL(n-1), FL(n-2)) with four calibration methods each. Expect 8–10 hours on a modern GPU or Apple Silicon.

**Step 5 — Find your results**

Once training finishes, generate the figures:

```bash
python scripts/generate_visualizations.py
```

Your results are in two places:
- `results/experiments/` — raw per-institution predictions (CSVs) and metrics (JSON)
- `results/figures/` — 6 publication-ready figures (heatmaps, reliability diagrams, ECE comparison)


---

## 🔬 Methodology

- **Data**: Freddie Mac mortgages (2006–2009), 14 institutions, 95 features after encoding
- **Default label**: Short Sale (code 03) or REO foreclosure (code 09)
- **Model**: 4-layer LSTM → 2 FC layers → logit output
- **FL algorithm**: FedAvg (McMahan et al., 2017), unweighted average
- **Calibration**: Temperature Scaling, Platt Scaling, Beta Calibration, Isotonic Regression
- **Metric**: ECE (Expected Calibration Error) — lower is better

---

## 📚 References

1. Lee, J., et al. (2023). *Federated Learning for Credit Risk Assessment*
2. McMahan, H.B., et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*
3. Guo, C., et al. (2017). *On Calibration of Modern Neural Networks*
4. Kull, M., et al. (2017). *Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers*

---


## 📜 License

MIT — see [LICENSE](LICENSE).
