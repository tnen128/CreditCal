# Quick Start Guide

## For Your Friends / Team

This guide helps you get started quickly with the FL Credit Risk Assessment framework.

## 🚀 Quick Setup (5 minutes)

### 1. Clone or Download

```bash
# If on GitHub
git clone https://github.com/yourusername/fl-credit-risk-assessment.git
cd fl-credit-risk-assessment

# If you received a ZIP file
unzip fl-credit-risk-assessment.zip
cd fl-credit-risk-assessment
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Get the Data

Download Freddie Mac data from: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

Place files in `data/raw/` directory:
```
data/raw/
├── sample_orig_2006.txt
├── sample_svcg_2006.txt
├── sample_orig_2007.txt
├── sample_svcg_2007.txt
...
```

## 🎯 Running Experiments

### Option 1: Quick Test (Recommended First)

**Uses only 2006Q1 data, 1 round, 1 epoch - completes in ~5 minutes**

```bash
# The config is already set for quick testing!
# Just run:
python src/evaluate_calibration.py
```

This will:
- Test all 5 scenarios
- Test all 4 calibration methods
- Generate all 6 visualization types
- Save results to `results/calibration/`

### Option 2: Full Experiment

**Uses all data 2006-2009, 100 rounds, 10 epochs - completes in 10-24 hours**

1. Edit `config/config.yaml`:
```yaml
# Change these lines:
training:
  global_rounds: 100  # Was 1
  local_epochs: 10    # Was 1

data:
  quick_test: false   # Was true
```

2. Run preprocessing:
```bash
python src/preprocess.py
```

3. Run full evaluation:
```bash
python src/evaluate_all_scenarios.py
python src/evaluate_calibration.py
```

## 📊 Viewing Results

After running, check these directories:

```
results/
├── evaluation/
│   └── evaluation_results.json     # Scenario comparison
└── calibration/
    ├── 1_heatmaps_comparison.png      # 3-panel heatmap
    ├── 2_reliability_diagrams_all.png # Reliability curves
    ├── 3_overall_performance.png      # Bar chart
    ├── 4_improvement_matrix.png       # ECE improvements
    ├── 5_method_ranking.png           # Best calibration methods
    ├── 6_brier_comparison.png         # Brier scores
    └── calibration_results_all.json   # All metrics
```

## 🔧 Adjusting Parameters

All configuration is in `config/config.yaml`. Common changes:

### Want Faster Testing?
```yaml
training:
  global_rounds: 1    # Fewer rounds
  local_epochs: 1     # Fewer epochs
  batch_size: 256     # Larger batches (if you have GPU memory)
```

### Want Better Results?
```yaml
training:
  global_rounds: 200  # More rounds
  local_epochs: 20    # More epochs per round
  learning_rate: 0.005  # Lower learning rate
```

### Want to Test Specific Scenarios?
```yaml
federated_learning:
  scenarios:
    local: true
    central: false      # Skip this one
    fl_n: true
    fl_n_minus_1: false  # Skip  this one
    fl_n_minus_2: false  # Skip this one
```

### Want Specific Calibration Methods?
```yaml
calibration:
  methods:
    platt: true
    isotonic: false  # Skip this one
    temperature: true
    beta: false      # Skip this one
```

## 📈 Understanding Results

### What Do the Metrics Mean?

**Classification Metrics**:
- **Accuracy**: % of correct predictions (higher = better)
- **F1 Score**: Balance between precision and recall (higher = better)
- **Precision**: % of positive predictions that are correct (higher = better)
- **Recall**: % of actual positives caught (higher = better)

**Calibration Metrics**:
- **ECE** (Expected Calibration Error): How well probabilities match reality (lower = better)
  - Good: < 0.05
  - Excellent: < 0.01
- **Brier Score**: Probability prediction accuracy (lower = better)
  - Good: < 0.1
  - Excellent: < 0.05

### Expected Results (2006Q1 Quick Test)

| Scenario | Accuracy | F1 | ECE (Before) | ECE (After Calib) |
|----------|----------|-----|--------------|-------------------|
| Local    | ~94%     | ~97%| ~0.32        | ~0.01             |
| Central  | ~92%     | ~96%| ~0.29        | ~0.007            |
| FL (n)   | ~92%     | ~96%| ~0.29        | ~0.006            |

## ❓ Troubleshooting

### Error: "No module named 'torch'"
```bash
pip install torch
```

### Error: "FileNotFoundError: data/raw/..."
Make sure you downloaded the Freddie Mac data and placed it in the correct directory.

### Error: "CUDA out of memory"
```yaml
# In config.yaml, reduce batch size:
training:
  batch_size: 64  # or even 32
```

### Slow Training?
- Use GPU if available (CUDA or MPS)
- Reduce data size (use quick_test mode)
- Reduce global_rounds and local_epochs

### Results Look Poor?
- Check if you're using enough rounds/epochs
- Verify data quality and preprocessing
- Try different learning rates
- Check model architecture

## 💡 Tips for Best Results

1. **Start Small**: Always run quick test first (1 round, 1 epoch)
2. **Monitor Progress**: Watch the console output for training loss
3. **Save Checkpoints**: Enable in config if training takes long
4. **Compare Scenarios**: Don't just run FL - compare all 5 scenarios!
5. **Calibrate**: Always run calibration analysis - it dramatically improves results
6. **Document Changes**: Keep notes on what parameters you changed

## 📚 Further Reading

- `docs/METHODOLOGY.md`: Detailed methodology
- `README.md`: Full project documentation
- Configuration comments in `config/config.yaml`
- Source code comments in `src/` directory

## 🆘 Getting Help

1. Check this guide first
2. Read the error message carefully
3. Check `config/config.yaml` for typos
4. Review console output for clues
5. Open an issue on GitHub (if applicable)

## 🎓 Learning Resources

If you're new to:
- **Federated Learning**: Read McMahan et al. (2017) - "Communication-Efficient Learning"
- **Calibration**: Read Guo et al. (2017) - "On Calibration of Modern Neural Networks"
- **Credit Risk**: Read the original Lee et al. (2023) paper

---

**Ready to go?** Start with the quick test:
```bash
python src/evaluate_calibration.py
```

Good luck! 🚀
