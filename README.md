# Glaucoma Health Economic Model

A comprehensive health economic model comparing AI-enhanced versus traditional (non-AI) glaucoma screening and management strategies.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)

## Overview

This model simulates the progression of glaucoma through four disease states (Mild, Moderate, Severe, and Visual Impairment) and evaluates the cost-effectiveness of different screening approaches over a 10-year time horizon using a Markov modeling framework.

## Key Features

- **Markov Model Structure**: Four-state transition model with annual cycles
- **Dual Screening Strategies**: AI-enhanced vs traditional screening comparison
- **Probabilistic Sensitivity Analysis**: 5,000 iterations for robust uncertainty analysis
- **Comprehensive Economic Evaluation**: Year-by-year breakdown of costs and QALYs

## Repository Structure

```
glaucoma-health-economic-model/
│
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── 01_model_setup_Portugal_costs.ipynb
│   ├── 01_model_setup_restructured.ipynb
│   ├── 01a_model_setup_Portugal_costs_utilities.ipynb
│   ├── 02_prevalence_sensitivity_rerun.ipynb
│   ├── 03_productivity_sensitivity_analysis.ipynb
│   ├── 04_starting_age_sensitivity_analysis.ipynb
│   ├── 05_ai_cost_sensitivity_analysis.ipynb
│   ├── 06_fp_disutility_sensitivity_analysis.ipynb
│   ├── 07_screening_frequency_sensitivity_analysis.ipynb
│   ├── 08_max_ai_cost_by_prevalence_optimized.ipynb
│   └── Mortality_loader.ipynb
│
├── src/                          # Source code directory
│   └── glaucoma_model/          # Core model package
│       ├── __pycache__/         # Python cache files
│       ├── __init__.py          # Package initialization
│       ├── analysis.py          # Analysis functions and utilities
│       ├── model.py             # Core Markov model implementation
│       ├── parameters.py        # Model parameters (costs, utilities, transitions)
│       ├── utils.py             # Helper functions and utilities
│       └── Mortality_loader.ipynb
│
└── README.md                     # This file
```

## Notebooks Directory

### Core Model

**01_model_setup_restructured.ipynb** - Foundation of the entire analysis
- Markov model structure definition (4 health states)
- Transition probability matrices for AI and non-AI strategies
- Cost parameters and health utility values
- Deterministic and probabilistic sensitivity analysis framework

**01_model_setup_Portugal_costs.ipynb** - Portuguese cost parameters variant
- Model setup with Portugal-specific cost data
- Country-specific healthcare cost estimates

**01a_model_setup_Portugal_costs_utilities.ipynb** - Extended Portuguese model
- Portuguese cost parameters with additional utility value considerations
- Comprehensive quality of life adjustments for Portuguese context

### Sensitivity Analyses

**02_prevalence_sensitivity_rerun.ipynb** - Impact of glaucoma prevalence variations on cost-effectiveness

**03_productivity_sensitivity_analysis.ipynb** - Evaluation of productivity loss impacts across disease states

**04_starting_age_sensitivity_analysis.ipynb** - Optimal screening initiation age analysis

**05_ai_cost_sensitivity_analysis.ipynb** - Robustness testing for AI screening cost variations

**06_fp_disutility_sensitivity_analysis.ipynb** - False positive rates and quality of life impacts

**07_screening_frequency_sensitivity_analysis.ipynb** - Comparison of different screening interval strategies

**08_max_ai_cost_by_prevalence_optimized.ipynb** - Advanced optimization: AI cost × prevalence interaction

**Mortality_loader.ipynb** - Load and process mortality data for model calibration

### Recommended Workflow

**For New Users**: Start with notebook 01_model_setup_restructured to understand the base model, then explore sensitivity analyses (02-07)

**For Researchers**: Review Mortality_loader.ipynb for data sources, examine notebook 01_model_setup_restructured for model structure, then run relevant sensitivity analyses

**For Policymakers**: Focus on notebooks 01_model_setup_restructured, 02, 04, and 05 for decision-relevant parameters. Country-specific versions (01_Portugal_costs and 01a_Portugal_costs_utilities) provide localized economic analysis.

## Model Structure

### Disease States
- **Mild Glaucoma** - Early stage with minimal vision loss
- **Moderate Glaucoma** - Progressive vision field defects
- **Severe Glaucoma** - Significant vision impairment
- **Visual Impairment (VI)** - Absorbing state with substantial vision loss

### Transition Probabilities

| Transition | AI Strategy | Non-AI Strategy |
|------------|-------------|-----------------|
| Mild → Moderate | 0.058 | 0.143 |
| Moderate → Severe | 0.040 | 0.087 |
| Severe → VI | 0.032 | 0.077 |

## Economic Parameters

### Annual Costs (EUR)
| State | Monitoring | Treatment | Productivity Loss |
|-------|------------|-----------|-------------------|
| Mild | €352 | €303 | €0 |
| Moderate | €463 | €429 | €0 |
| Severe | €644 | €609 | €0 |
| VI | €576 | €662 | €7,630 |


### Health Utilities (0-1 scale)
- **Mild**: 0.985 ± 0.023
- **Moderate**: 0.899 ± 0.039
- **Severe**: 0.773 ± 0.046
- **VI**: 0.634 ± 0.052

### Diagnostic Accuracy Parameters

The model uses **sensitivity** and **specificity** values from the Parameter class to characterize screening performance for both AI and non-AI strategies. 

**Important Note**: True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN) are **not used** in the analysis. All diagnostic accuracy calculations are derived solely from the sensitivity and specificity parameters. In addition, the Parameter class serves as a backbone. Not all parameters are used and some are changed in the various notebooks. 

## Installation

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scipy jupyter
```

### Clone Repository
```bash
git clone https://github.com/felizzi/Glaucoma_Studies.git
cd Glaucoma_Studies
```

## Usage

### Quick Start
```python
from glaucoma_model.model import AIGlaucomaModel, NonAIGlaucomaModel

# Initialize models
ai_model = AIGlaucomaModel()
non_ai_model = NonAIGlaucomaModel()

# Define initial population distribution
initial_dist = {
    'Mild': 0.704,
    'Moderate': 0.167,
    'Severe': 0.130,
    'VI': 0.0
}

# Run deterministic analysis
ai_results = ai_model.run_deterministic(initial_dist=initial_dist)
non_ai_results = non_ai_model.run_deterministic(initial_dist=initial_dist)
```

### Probabilistic Sensitivity Analysis
```python
# Run probabilistic analysis (5,000 iterations)
ai_psa_results = ai_model.run_probabilistic(
    n_iterations=5000, 
    initial_dist=initial_dist
)
non_ai_psa_results = non_ai_model.run_probabilistic(
    n_iterations=5000, 
    initial_dist=initial_dist
)
```

## Applications

- **Health Technology Assessment** of AI glaucoma screening programs
- **Resource Allocation** decisions in ophthalmology services
- **Cost-Effectiveness Analysis** for healthcare policymakers
- **Budget Impact Analysis** for implementing AI screening

## Model Limitations

- Assumes perfect adherence to screening and treatment protocols
- Simplified four-state disease progression structure
- Based on specific cost and utility estimates (may vary by healthcare system)
- Limited to 10-year time horizon
- Does not account for patient heterogeneity beyond modeled disease states

## Citation

If you use this model in your research or policy analysis, please cite:

```bibtex
@misc{felizzi_glaucoma_studies,
  author       = {Felizzi, Federico},
  title        = {Glaucoma\_Studies},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/felizzi/Glaucoma_Studies}},
  note         = {Accessed: 2025-12-13}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**GitHub Issues**: [Create an issue](https://github.com/felizzi/Glaucoma_Studies/issues)

---

**Disclaimer**: This model is for research and educational purposes. Clinical and policy decisions should involve appropriate medical and economic expertise.
