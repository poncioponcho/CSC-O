# CSC-O: Causal-Stratified Counterfactual Optimization for Antibody CDR3 Design

> **Version**: rf3-optimization-v1 (First Release)
> **Date**: 2026-06-01
> **Full Name**: Causal-Stratified Counterfactual Optimization Pipeline

---

## Overview

CSC-O is a computational pipeline for antibody CDR3 region design optimization. It integrates **causal inference**, **stratified survival analysis**, and **counterfactual reasoning** to derive actionable design strategies from high-throughput antibody sequence data.

Given 10,572 antibody sequences targeting the Q02223 epitope, the pipeline:
1. Extracts CDR3 sequence features and constructs multi-stage funnel survival data
2. Performs stratified attribution via Cox regression and Kaplan-Meier analysis
3. Estimates causal effects (ATE/CATE) of sequence-level interventions
4. Generates counterfactual mutation suggestions with individual-level confidence
5. Synthesizes design rules into a machine-readable strategy configuration
6. Generates and screens novel CDR3 candidates for experimental validation

---

## rf3 Optimization Version — Key Improvements

This release represents the first rf3 optimization version, incorporating three major research directions beyond the baseline pipeline:

### Direction 1: ESM-2 Multi-Scale Fusion Embedding

| Feature | Baseline (v1.0) | rf3 Optimization |
|---------|----------------|------------------|
| Model | esm2_t12_35M_UR50D only | t12 + t30 dual-model fusion |
| Embedding Dim | 480 | 1120 (480 + 640 concat) |
| Fusion Strategy | N/A | concat / average / pca_concat |

- Supports configurable multi-model ESM-2 encoding via `--esm2-models`
- Three fusion strategies: concatenation, weighted averaging, PCA-reduced concatenation
- Per-model `.npy` caching for incremental computation
- Model registry with 4 ESM-2 variants (t12, t30, t33, t36)

### Direction 2: Heterogeneous CATE + Subgroup Discovery

| Feature | Baseline (v1.0) | rf3 Optimization |
|---------|----------------|------------------|
| CATE Method | Constant θ (Double ML) | CausalForestDML → R-learner → PLR cascade |
| Subgroup Analysis | None | K-means clustering on CATE landscape |
| Individual CATE | No | Per-sequence CATE + SE + significance |

- Three-method cascade: CausalForestDML (preferred) → R-learner (fallback) → Partially Linear Regression (last resort)
- R-learner with **propensity score clipping** (0.1–0.9) and **CATE clipping** (±50) for numerical stability
- Automatic discrete/continuous treatment detection
- K-means subgroup discovery with profile extraction
- Multi-treatment causal model with enhanced confounders

### Direction 3: Constrained Sequence Generator + Fast Screener

| Feature | Baseline (v1.0) | rf3 Optimization |
|---------|----------------|------------------|
| Sequence Generation | None | Monte Carlo constrained sampler |
| Screening | None | GBC-based fast screener |
| Output | Analysis report only | Top-200 candidate sequences |

- **Generator** (`csco_generator.py`): Monte Carlo CDR3 sampler with hard constraints, soft preferences, anti-pattern filtering, template-guided mutation (30%)
- **Screener** (`csco_screener.py`): GradientBoostingClassifier on 12 CDR3-derived features (not ESM-2 embeddings, avoiding overfitting)
- Diversity filtering via minimum edit distance
- Screening AUC: 0.895; predicted pass rate: 54.9%

---

## Pipeline Architecture

```
Input: Antibody sequence CSV (10,572 sequences)
    │
    ├─ Stage 1: Data Engineering
    │   ├─ CDR3 feature extraction (length, amino acid ratios, patterns)
    │   └─ Multi-stage funnel survival data construction
    │
    ├─ Stage 2: Stratified Attribution
    │   ├─ Cox proportional hazards regression
    │   ├─ Kaplan-Meier survival curves
    │   └─ Threshold sensitivity analysis
    │
    ├─ Stage 3a: Causal Inference Engine
    │   ├─ PC algorithm causal DAG discovery
    │   └─ ATE estimation (Double ML)
    │
    ├─ Stage 3b: ESM-2 Sequence Encoding
    │   ├─ Multi-model encoding (t12 + t30)
    │   └─ Fusion embedding (concat/average/pca_concat)
    │
    ├─ Stage 3c: Counterfactual Navigation
    │   ├─ Heterogeneous CATE (CausalForestDML / R-learner / PLR)
    │   ├─ K-means subgroup discovery
    │   └─ Individual-level mutation suggestions
    │
    ├─ Stage 4: Sequence Generation & Screening
    │   ├─ Constrained Monte Carlo CDR3 sampler (10K candidates)
    │   └─ GBC fast screener → Top-200 candidates
    │
    └─ Stage 5: Rule Synthesis
        ├─ design_strategy.json (machine-readable)
        ├─ csco_analysis_report.txt (human-readable)
        └─ Comprehensive analysis report
```

---

## Project Structure

```
analyze1/
├── csco_pipeline.py              # Main pipeline entry point (6 stages)
├── csco_data_engineering.py      # Stage 1: Feature extraction
├── csco_layer1_stratified.py     # Stage 2: Stratified analysis
├── csco_layer2_causal.py         # Stage 3a: Causal inference
├── csco_esm2_encode.py           # Stage 3b: ESM-2 encoding (standalone)
├── csco_layer3_counterfactual.py # Stage 3c: Counterfactual navigation
├── csco_layer3_analysis.py       # Counterfactual analysis (standalone)
├── csco_layer5_synthesis.py      # Stage 5: Rule synthesis
├── csco_visualize.py             # Visualization dashboard
├── csco_generator.py             # Constrained CDR3 sequence generator
├── csco_screener.py              # Fast sequence screening pipeline
├── requirements.txt              # Python dependencies
├── PROJECT_PROGRESS_AND_PLAN.md  # Project progress documentation
├── docs/
│   └── 4090_vpn_remote_run_guide.html
├── tools/                        # Server management utilities
│   ├── vpn_watchdog.sh           # VPN connection monitor
│   ├── ssh_auto_attach.sh        # SSH auto-reconnect
│   ├── check_server_status.sh    # Server status checker
│   ├── workspace_save.py         # tmux session persistence
│   ├── workspace_restore.py      # tmux session restoration
│   ├── vim_session_save.sh       # Vim state persistence
│   ├── vim_session_restore.sh    # Vim state restoration
│   └── install.sh                # Dependency installer
├── output_server/                # v1.0 baseline results
│   ├── csco_analysis_report.txt
│   ├── design_strategy.json
│   ├── CSC-O_实验分析报告.md
│   ├── 云端算力增强实验方案.md
│   └── 方案A+B数学原理详解.md
└── output_server_v2/             # rf3 optimization results
    ├── csco_analysis_report.txt
    └── design_strategy.json
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for multi-model ESM-2)

### Installation

```bash
# Clone repository
git clone https://github.com/poncioponcho/CSC-O.git
cd CSC-O/analyze1

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For heterogeneous CATE estimation (optional)
pip install econml
```

### Running the Pipeline

```bash
# Basic run (baseline configuration)
python csco_pipeline.py \
  --input data/Q02223_first50_all_sequences.csv \
  --output output \
  --work work \
  --device cuda

# rf3 optimization run (multi-model ESM-2 + R-learner + subgroups)
python csco_pipeline.py \
  --input data/Q02223_first50_all_sequences.csv \
  --output output \
  --work work \
  --device cuda \
  --esm2-models esm2_t12_35M_UR50D esm2_t30_150M_UR50D \
  --esm2-fusion concat \
  --cate-method r_learner \
  --subgroup-clusters 3

# Generate candidate sequences
python csco_generator.py \
  --strategy-file output/design_strategy.json \
  --n-samples 10000 \
  --output output/generated_sequences.csv

# Screen candidates
python csco_screener.py \
  --input output/generated_sequences.csv \
  --training-data data/Q02223_first50_all_sequences.csv \
  --top-n 200 \
  --output output/screened_candidates.csv
```

### Key CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--esm2-models` | `esm2_t12_35M_UR50D` | ESM-2 model(s) for encoding |
| `--esm2-fusion` | `concat` | Fusion strategy: concat, average, pca_concat |
| `--esm2-pca-dim` | `128` | PCA dimension for pca_concat fusion |
| `--cate-method` | `causal_forest` | CATE method: causal_forest, r_learner, plr |
| `--subgroup-clusters` | `3` | Number of K-means subgroups |
| `--device` | `cuda` | Device: cuda or cpu |
| `--resume` | False | Resume from last checkpoint |

---

## Key Experimental Results

### Baseline (v1.0)

| Metric | Value |
|--------|-------|
| Total sequences | 10,572 |
| RF2 pass rate | 11.7% |
| Final candidate rate | 0.61% |
| CDR3 allowed lengths | [5, 6, 7] |
| Top salvage edit | Pos0 G→Y (PAE Δ = -1.85) |

### rf3 Optimization

| Metric | Baseline | rf3 Optimization |
|--------|----------|------------------|
| Embedding dimension | 480 | 1120 (t12+t30 concat) |
| CATE method | Constant θ | Heterogeneous θ(x) |
| CATE range | N/A | [-28.8, 31.2] (R-learner, clipped) |
| Subgroup discovery | None | 3 subgroups (3255 / 6382 / 935) |
| Sequence generation | 0 | 10,000 candidates/round |
| Screening AUC | N/A | 0.895 |
| Predicted pass rate | 11.7% | 54.9% (screened candidates) |

### Critical Design Rules (from Causal Analysis)

1. **First residue must be F/V/W/Y** — ATE = -6.54 on PAE (strongest protective effect)
2. **Avoid serine ratio >15%** — HR = 1.77, ATE = +9.69 (strongest risk factor)
3. **CDR3 length 6–7** — Length ≥10 drops pass rate from 30.8% to 2.6%

---

## Important Notes

### Data Privacy
- All sensitive information (server IPs, credentials, VPN details, local paths) has been redacted and replaced with placeholders (`REDACTED_IP`, `REDACTED_USER`, etc.)
- CSV data files are excluded from the repository via `.gitignore`
- No personally identifiable information is stored in the codebase

### Known Limitations
1. **Treatment variable selection**: Using `rf2_passed` (outcome) as treatment violates the causal manipulability assumption; future versions should use sequence-level features as treatments
2. **Optimization target**: Current pipeline optimizes RF2 pass rate, not `final_candidate` probability; RF2 is only the first gate in the multi-stage funnel (RF2 → AF3 → Schrödinger → final_candidate at 0.61%)
3. **R-learner numerical instability**: Extreme treatment imbalance (11.7% pass rate) can cause CATE explosion; mitigated by propensity/CATE clipping but not fundamentally resolved
4. **Screener generalization**: GBC screener trained on 12 CDR3-derived features achieves AUC 0.895; performance on out-of-distribution sequences is unvalidated

### Computational Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1× GTX 1080 Ti (11GB) | 2× GPU for multi-model ESM-2 |
| RAM | 8GB | 16GB+ |
| Storage | 5GB | 20GB+ (for ESM-2 model weights) |
| Runtime (baseline) | ~6-8 min | — |
| Runtime (rf3 optimization) | ~25-30 min | — |

---

## Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
torch>=1.10.0
fair-esm>=2.0.0
lifelines>=0.27.0
lightgbm>=3.3.0
openpyxl>=3.0.0
econml>=0.14.0  # Optional, for CausalForestDML
```

**Note**: If using PyTorch with CUDA, ensure compatible versions:
- CUDA 11.x → `torch>=1.10.0` with corresponding CUDA toolkit
- Known compatible: `numpy<1.24` with `torch 1.12.1`

---

## License

This project is for research purposes only. Please contact the authors for commercial use.

---

## Citation

If you use this pipeline in your research, please cite:

```
CSC-O: Causal-Stratified Counterfactual Optimization for Antibody CDR3 Design
Version: rf3-optimization-v1, 2026
```
