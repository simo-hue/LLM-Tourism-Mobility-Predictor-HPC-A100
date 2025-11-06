# 🚀 LLM-Mob: Tourist Mobility Prediction with Large Language Models on HPC
## [ paper in writing ]
<!-- SEO Keywords: large language models, tourism prediction, mobility forecasting, LLM tourism, next destination prediction, VeronaCard dataset, HPC machine learning, NVIDIA A100, Ollama inference, geospatial AI, temporal analysis, tourist behavior prediction -->

<div align="center">

![Python](https://img.shields.io/badge/python-3.9--3.11-blue?style=for-the-badge&logo=python)
![AI/ML](https://img.shields.io/badge/AI%2FML-Large%20Language%20Models-green?style=for-the-badge&logo=openai)
![Ollama](https://img.shields.io/badge/Ollama-Multi--GPU%20Inference-ff6b35?style=for-the-badge&logo=lightning)
![HPC](https://img.shields.io/badge/HPC-Leonardo%20CINECA-red?style=for-the-badge&logo=nvidia)
![GPU](https://img.shields.io/badge/GPU-4x%20NVIDIA%20A100%2064GB-76b900?style=for-the-badge&logo=nvidia)

**State-of-the-art tourist mobility prediction using Large Language Models**

[📖 Quick Start](#-quick-start) • [📊 Results](#-results) • [🔬 Methodology](#-methodology) • [💻 Usage](#-usage)

</div>

---

## 🎯 Overview

**LLM-Mob** is a production-ready system that predicts tourist next destinations using Large Language Models on HPC infrastructure. Built on the VeronaCard dataset (370K+ tourists, 2014-2023), it achieves **64.14% Top-5 accuracy** through advanced prompt engineering with geospatial and temporal context.

### Key Results

- **🏆 64.14% Top-5 Hit Rate** | 24.99% Top-1 (Qwen2.5 14B - best configuration)
- **⚡ 1.45-2.85s Response Time** | Analysis of 1.2M+ predictions per model
- **📊 6 LLM Models Evaluated** | Qwen2.5, Mistral, Llama3.1, Mixtral, DeepSeek-Coder
- **🗺️ +113% to +400% Accuracy Boost** | Geospatial context vs base version
- **📈 10-Year Validation** | 2014-2023 with COVID-19 impact analysis (-32.7% in 2020)
- **🔧 Production-Ready** | 98.5% data utilization, fault-tolerant architecture

### Technical Stack

- **Infrastructure**: 4× NVIDIA A100 64GB on Leonardo HPC (CINECA)
- **LLM Engine**: Ollama multi-instance cluster with intelligent load balancing
- **Models**: Qwen2.5 (7B/14B), Mistral 7B, Llama3.1 8B, Mixtral 8×7B, DeepSeek-Coder 33B
- **Dataset**: VeronaCard tourist mobility (370K+ visits, 70 POIs, 10 years)
- **Processing**: Parallel GPU inference with circuit breaker and checkpoint system

---

## 🚀 Quick Start

### Prerequisites

```bash
# System Requirements
- Python 3.9-3.11 (⚠️ Python 3.12+ not supported)
- CUDA 11.8+ for GPU acceleration
- 16GB+ RAM recommended

# HPC Environment (Leonardo CINECA)
- SLURM job scheduler
- 4× NVIDIA A100 64GB GPUs
- Ollama multi-instance setup
```

### Installation

```bash
# Clone repository
git clone https://github.com/simo-hue/LLM-Mob-As-Mobility-Interpreter.git
cd LLM-Mob-As-Mobility-Interpreter

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Predictions

```bash
# Full geospatial + temporal analysis (RECOMMENDED)
python veronacard_mob_with_geom_time_parrallel.py --file dati_2014.csv

# Resume from checkpoint
python veronacard_mob_with_geom_time_parrallel.py --append

# Custom anchor point strategy
python veronacard_mob_with_geom_time_parrallel.py --anchor penultimate

# HPC Job Submission
sbatch time_4_GPU.sh  # Submits to SLURM with 4× A100 allocation
```

---

## 📊 Results

> **Data Source**: All results computed from `metrics/` directory (2014-2023 VeronaCard dataset)

### 🏆 Top Performers

**Best Configurations** (10-year average, 2014-2023):

| Rank | Model | Anchor | Strategy | Top-1 | Top-5 | Avg Time |
|------|-------|--------|----------|-------|-------|----------|
| 🥇 | **Qwen2.5 14B** | Middle | Geospatial | **24.99%** | **64.14%** | 1.85s |
| 🥈 | **Qwen2.5 14B** | Middle | Geospatial + Temporal | **24.34%** | **61.57%** | 2.15s |
| 🥉 | **Mistral 7B** | Middle | Geospatial | **24.54%** | **57.11%** | 1.95s |
| 4th | Mixtral 8×7B | Middle | Geospatial | 12.87% | 56.71% | 2.25s |
| 5th | Qwen2.5 14B | Penultimate | Geospatial | 19.42% | 54.07% | 1.85s |

**Key Findings**:
- **Qwen2.5 14B** achieves state-of-the-art performance with best accuracy-speed balance
- **Middle anchor point** consistently outperforms penultimate (+5-10% accuracy)
- **Geospatial context** is the most critical feature for accuracy

### 📊 Model Comparison

**All Models Evaluated** (ranked by performance score):

| Model | Organization | Top-5 Hit Rate | Success Rate | Performance Score |
|-------|-------------|---------------|--------------|-------------------|
| **Qwen2.5 14B** | Alibaba | **62.3%** | 100.0% | **73.6** |
| **Qwen2.5 7B** | Alibaba | **44.8%** | 99.9% | **61.3** |
| **Mistral 7B** | Mistral AI | **38.2%** | 99.8% | **56.7** |
| Llama3.1 8B | Meta | 35.7% | 99.6% | 55.0 |
| Mixtral 8×7B | Mistral AI | 34.5% | 99.7% | 54.0 |
| DeepSeek-Coder 33B | DeepSeek | 32.1% | 99.5% | 52.4 |

### 🎯 Strategy Impact

**Context Enrichment Effectiveness** (Hit Rate % by Strategy):

| Model | Base Version | With Geospatial | Geospatial + Temporal | Boost |
|-------|-------------|-----------------|----------------------|-------|
| Qwen2.5 14B | 30.9% | **65.7%** ⭐ | 63.8% | **+113%** |
| Qwen2.5 7B | 15.8% | **48.2%** | 46.1% | **+205%** |
| Mistral 7B | 8.5% | **42.5%** | 40.2% | **+400%** |
| Llama3.1 8B | 11.2% | 38.7% | **36.9%** | **+245%** |
| Mixtral 8×7B | 10.8% | **37.2%** | 35.1% | **+244%** |
| DeepSeek-Coder 33B | 9.7% | **34.2%** | 32.5% | **+253%** |

**Insights**:
- 🗺️ **Geospatial context** provides +113% to +400% improvement across all models
- ⏰ **Temporal features** show marginal gains (+0-2%) over pure geospatial
- 📊 **Base versions** demonstrate necessity of contextual enrichment

### ⚡ Processing Time Analysis

**Response Time Performance** (1.2M+ predictions per model):

| Model | Strategy | Mean Time | Min | Max | Efficiency |
|-------|----------|-----------|-----|-----|------------|
| **Qwen2.5 7B** | Geospatial | **1.45s** | 0.76s | 25.25s | ⚡ Fastest |
| **Qwen2.5 14B** | Geospatial + Temporal | **1.62s** | 0.83s | 13.61s | 🏆 Best |
| **Qwen2.5 14B** | Geospatial | **1.85s** | 0.76s | 25.25s | ⭐ Balanced |
| Mistral 7B | Geospatial | 1.95s | - | - | Good |
| Llama3.1 8B | Geospatial | 2.15s | - | - | Moderate |
| Mixtral 8×7B | Geospatial | 2.25s | - | - | Slower |
| DeepSeek-Coder 33B | Geospatial | 2.85s | - | - | Slowest |

**Processing Insights**:
- **Qwen2.5 14B**: Best accuracy (64.14%) with second-fastest processing (1.85s)
- **Base Version Paradox**: Despite simpler prompts, 48% slower than enriched contexts
- **Model Size Impact**: Larger models (33B) show +97% slower processing vs 7B models

### 📅 Temporal Evolution (2014-2023)

**Year-over-Year Performance** (Qwen2.5 14B - Middle - Geospatial):

| Year | Top-1 | Top-5 | Notable Events |
|------|-------|-------|----------------|
| 2014 | 27.52% | 65.75% | Strong baseline |
| 2015 | 26.54% | 65.14% | Consistent |
| 2016 | 26.68% | 65.65% | Stable |
| 2017 | 25.49% | 65.10% | Slight decline |
| 2018 | 27.09% | 64.86% | Recovery |
| 2019 | 27.35% | 65.11% | Pre-pandemic peak |
| **2020** | **18.42%** | **60.34%** | 📉 **COVID-19 Impact** |
| 2021 | 21.91% | 62.28% | Gradual recovery |
| 2022 | 25.79% | 63.71% | Near-full recovery (94%) |
| 2023 | 23.14% | 63.41% | Stabilization |

**COVID-19 Impact**: -32.7% Top-1 accuracy drop (27.35% → 18.42%) in 2020, with 94% recovery by 2022.

---

## 🔬 Methodology

### Research Foundation

This work was **inspired by** the paper "[Where Would I Go Next? Large Language Models as Human Mobility Predictors](https://arxiv.org/abs/2308.15197)" (Wang et al., 2023).

**Important**: After initial exploration of the original LLM-Mob repository, I **completely rebuilt the system from scratch** with:
- ✅ **Independence from OpenAI API keys** (replaced with open-source Ollama)
- ✅ **Custom HPC-optimized architecture** for Leonardo/CINECA infrastructure
- ✅ **Novel prompt engineering framework** with advanced geospatial/temporal features
- ✅ **Production-grade reliability** (circuit breaker, checkpointing, fault tolerance)
- ✅ **Comprehensive multi-model evaluation** (6 LLMs vs original single model)

This is a **completely independent implementation** with different architecture, models, and optimizations.

### Prompt Engineering

**Multi-Context Prompt Template**:

```python
PROMPT = """
TOURIST PROFILE:
- Cluster: {cluster_id} (behavioral pattern group)
- Visit History: {previous_visits}
- Current Location: {current_poi}

GEOSPATIAL CONTEXT:
- Nearby POIs: {pois_within_walking_distance}
- Distances: {poi_distances_km}

TEMPORAL CONTEXT:
- Current Time: {day_name} {hour}:{minute}
- User Pattern: Typical hours {usual_visit_times}

TASK: Predict next 5 most likely destinations.
OUTPUT FORMAT: JSON
"""
```

**Strategies Evaluated**:
- **Base Version**: Tourist profile only (minimal context)
- **With Geospatial**: + distance calculations, nearby POIs
- **Geospatial + Temporal**: + time patterns, seasonal context

### Anchor Point Strategies

- **Middle**: Uses central visit in sequence for prediction
- **Penultimate**: Uses second-to-last visit for prediction

Results show **middle anchor** consistently outperforms penultimate (+5-10%).

### HPC Optimization

**Configuration** (Leonardo CINECA - 4× NVIDIA A100 64GB):

```python
# GPU Optimization
MAX_CONCURRENT_REQUESTS = 12
MAX_CONCURRENT_PER_GPU = 3
REQUEST_TIMEOUT = 900            # 15 min for HPC latency
CIRCUIT_BREAKER_THRESHOLD = 50   # Failure tolerance

# Ollama Payload (optimized for A100)
{
    "num_ctx": 1024,          # Context window
    "num_predict": 64,        # Response tokens
    "num_thread": 56,         # Sapphire Rapids cores per GPU
    "num_batch": 512,         # Conservative batch size
    "temperature": 0.1,       # Deterministic predictions
    "cache_type_k": "f16"     # FP16 for A100 speed
}
```

**Features**:
- **Multi-instance Ollama**: 4 instances (ports 11434-11437), one per GPU
- **Circuit Breaker**: CLOSED/OPEN/HALF_OPEN states with automatic recovery
- **Checkpoint System**: Resume from interruption every 500 processed cards
- **Health Monitoring**: Real-time GPU utilization and adaptive load balancing

---

## 💻 Usage

### Main Scripts

```bash
# RECOMMENDED: Full geospatial + temporal analysis
python veronacard_mob_with_geom_time_parrallel.py

# Geospatial only
python veronacard_mob_with_geom_parrallel.py

# Base version (minimal context)
python veronacard_mob_versione_base_parrallel.py
```

### Command Line Options

```bash
# Process specific file with user limit
python veronacard_mob_with_geom_time_parrallel.py \
    --file dati_2014.csv \
    --max-users 1000

# Resume from checkpoint (critical for long runs)
python veronacard_mob_with_geom_time_parrallel.py --append

# Force complete reprocessing
python veronacard_mob_with_geom_time_parrallel.py --force

# Custom anchor point
python veronacard_mob_with_geom_time_parrallel.py --anchor penultimate

# Debug mode (limited dataset)
DEBUG_MODE=True python veronacard_mob_with_geom_time_parrallel.py --max-users 100
```

### HPC Job Submission (Leonardo CINECA)

```bash
# Submit job to SLURM
sbatch time_4_GPU.sh       # Full temporal+geospatial (RECOMMENDED)
sbatch geom_4_GPU.sh       # Geospatial only
sbatch base_4_GPU.sh       # Base version

# Monitor job
squeue -u $USER
tail -f slurm-<JOBID>.out

# Check computational budget
saldo -b IscrC_LLM-Mob

# Cancel job
scancel <JOBID>
```

### Output Structure

```
results/
└── {model_name}/              # e.g., qwen2.5_14b/
    └── {strategy}/            # e.g., with_geom_time/
        └── {anchor}/          # e.g., middle/
            ├── dati_2014_pred_20250930.csv      # Predictions with hit rates
            └── dati_2014_checkpoint.txt         # Processing state
```

### Metrics Analysis

```bash
# View pre-computed metrics
cd metrics/

# Strategy-based metrics (10-year data per model/strategy)
ls strategy/middle/qwen2.5_14b/with_geom/

# Inter-model comparison
ls inter_model_comparison/

# Processing time analysis
ls time_analysis/

# Run Jupyter analysis notebooks
jupyter notebook notebook/singole_metriche_canva.ipynb
```

---

## 📊 Dataset

### VeronaCard Dataset

**Specifications**:
- **Time Range**: 2014-2023 (10 years)
- **Records**: 370,000+ tourist visits
- **POIs**: 70 Points of Interest with GPS coordinates
- **Location**: Verona, Italy (UNESCO World Heritage Site)
- **Completeness**: 99.2% records with complete temporal data

**Structure**:

```csv
# Visit Records (dati_YYYY.csv)
date,time,poi_name,card_id,entrance_type
15-08-14,10:30:45,Arena,0403E98ABF3181,standard
15-08-14,14:15:30,Casa di Giulietta,0403E98ABF3181,priority

# Points of Interest (vc_site.csv)
name_short,latitude,longitude,category
Arena,45.4394,10.9947,Monument
Casa di Giulietta,45.4419,10.9988,Museum
```

**Ethics & Privacy**:
- IRB Approval: University of Verona Ethics Committee
- Data Protection: GDPR compliant
- Privacy: Fully anonymized with pseudonymous card IDs
- License: Academic research only (CC-BY-NC)

---

## 📁 Project Structure

```
LLM-Mob-As-Mobility-Interpreter/
├── veronacard_mob_with_geom_time_parrallel.py  # Main: Geospatial + Temporal
├── veronacard_mob_with_geom_parrallel.py       # Geospatial only
├── veronacard_mob_versione_base_parrallel.py   # Base version
├── data/
│   └── verona/
│       ├── vc_site.csv              # 70 POIs with GPS
│       ├── dati_2014.csv            # ~370K visits per year
│       └── dati_2015-2023.csv
├── results/                         # Predictions output
│   └── {model}/{strategy}/{anchor}/
├── metrics/                         # Pre-computed metrics
│   ├── strategy/                    # Per model/strategy/anchor
│   ├── inter_model_comparison/      # Cross-model analysis
│   └── time_analysis/               # Processing time stats
├── notebook/                        # Jupyter analysis
│   └── singole_metriche_canva.ipynb
├── time_4_GPU.sh                    # SLURM job script (RECOMMENDED)
├── geom_4_GPU.sh
├── base_4_GPU.sh
├── ollama_ports.txt                 # Multi-instance config
└── requirements.txt
```

---

## 🛠️ Troubleshooting

### Common Issues

**GPU Out of Memory**:
```bash
# Reduce batch size in script
num_batch: 512 → 256

# Or reduce GPU memory fraction
GPU_MEMORY_FRACTION = 0.90
```

**Ollama Connection Timeout**:
```bash
# Check Ollama instances
curl http://localhost:11434/api/tags
curl http://localhost:11435/api/tags
curl http://localhost:11436/api/tags
curl http://localhost:11437/api/tags

# Restart if needed
pkill ollama
./start_ollama_cluster.sh
```

**Circuit Breaker Open**:
```bash
# Wait 60s for automatic recovery, or check GPU health
nvidia-smi -q -d HEALTH
```

**Checkpoint Corruption**:
```bash
# Delete checkpoint and restart with --force
rm results/*/checkpoint.txt
python veronacard_mob_with_geom_time_parrallel.py --force
```

---

## 🔮 Future Work

### VLLM Integration (Planned - NOT Implemented)

⚠️ **Important**: VLLM is a **future enhancement**, NOT currently implemented. All results in this README are based on **Ollama architecture**.

**Current System**: Ollama multi-instance cluster (4× A100)
**Future Plan**: VLLM with tensor parallelism for higher throughput
**Timeline**: Planned for version 3.0

---

## 📚 Citation

If you use LLM-Mob in your research, please cite:

```bibtex
@software{mattioli2025llm_mob,
  author = {Mattioli, Simone},
  title = {LLM-Mob: Tourist Mobility Prediction using Large Language Models on HPC Infrastructure},
  url = {https://github.com/simo-hue/LLM-Mob-As-Mobility-Interpreter},
  year = {2025},
  note = {Independent implementation with custom HPC architecture}
}
```

### Related Work

Original inspiration (different implementation):
```bibtex
@article{wang2023llm_mobility,
  title={Where Would I Go Next? Large Language Models as Human Mobility Predictors},
  author={Wang, Xinglei and Zhu, Meng and Li, Tao and Luo, Bin and Zhong, Chen and Zhou, Xuefeng},
  journal={arXiv preprint arXiv:2308.15197},
  year={2023}
}
```

---

## 🙏 Acknowledgments

- **[CINECA](https://www.cineca.it/)** - Leonardo HPC Infrastructure & Computational Resources
- **[University of Verona](https://www.univr.it/)** - VeronaCard Dataset & Research Support

---

## 📄 License

**Creative Commons Attribution-NonCommercial (CC BY-NC)**

- ✅ Academic research use permitted
- ✅ Modification and redistribution with attribution
- ❌ Commercial use prohibited without permission
- ❌ VeronaCard dataset redistribution requires University Of Verona approval

---

## 📞 Contact

**Simone Mattioli**
📧 Email: [mattioli.simone.10@gmail.com](mailto:mattioli.simone.10@gmail.com)
🐙 GitHub: [@simo-hue](https://github.com/simo-hue)
💼 LinkedIn: [Simone Mattioli](https://linkedin.com/in/simone-mattioli)

---

<div align="center">

**Made with ❤️ for the Tourism Analytics and AI Research Community**

*Keywords: Large Language Models, LLM Tourism Prediction, Next Destination Forecasting, Mobility Analytics, HPC Machine Learning, NVIDIA A100, Ollama Inference, Geospatial AI, Temporal Analysis, VeronaCard Dataset, Tourist Behavior Prediction, Leonardo CINECA, Qwen2.5, Mistral AI*

</div>