# Glaucoma Health Economic Model

A comprehensive health economic model comparing AI-enhanced versus traditional (non-AI) glaucoma screening and management strategies.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)

## Overview

This model simulates the progression of glaucoma through four disease states (Mild, Moderate, Severe, and Visual Impairment) and evaluates the cost-effectiveness of different screening approaches over a 10-year time horizon using a Markov modeling framework.

## 🔬 Key Features

- **Markov Model Structure**: Four-state transition model with annual cycles
- **Dual Screening Strategies**:
  - AI-enhanced screening with early detection and comprehensive care
  - Traditional screening with late detection (only VI cases receive treatment)
- **Probabilistic Sensitivity Analysis**: 5,000 iterations for robust uncertainty analysis
- **Comprehensive Economic Evaluation**: Year-by-year breakdown of costs and QALYs
- **Multiple Scenarios**: General population and diabetic retinopathy screening

## 🏥 Model Structure

### Disease States
- **Mild Glaucoma** - Early stage with minimal vision loss
- **Moderate Glaucoma** - Progressive vision field defects
- **Severe Glaucoma** - Significant vision impairment
- **Visual Impairment (VI)** - Absorbing state with substantial vision loss

### Transition Probabilities
The model incorporates different progression rates based on screening strategy:

| Transition | AI Strategy | Non-AI Strategy |
|------------|-------------|-----------------|
| Mild → Moderate | 0.058 | 0.143 |
| Moderate → Severe | 0.040 | 0.087 |
| Severe → VI | 0.032 | 0.077 |

## 💰 Economic Parameters

### Annual Costs (USD)
| State | Monitoring | Treatment | Productivity Loss |
|-------|------------|-----------|-------------------|
| Mild | €352 | €303 | €0 |
| Moderate | €463 | €429 | €0 |
| Severe | €644 | €609 | €0 |
| VI | €576 | €662 | €7,630 |

### Screening Costs
- **AI Screening**: $11.50 per patient
- **Traditional Screening**: $100.00 per patient

### Health Utilities (0-1 scale)
- **Mild**: 0.985 ± 0.023
- **Moderate**: 0.899 ± 0.039
- **Severe**: 0.773 ± 0.046
- **VI**: 0.634 ± 0.052

### Screening Performance
- **AI Strategy**: Sensitivity 77.5%, Specificity 95.4%
- **Traditional Strategy**: Lower performance metrics

## 🛠️ Installation

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scipy jupyter
```

### Clone Repository
```bash
git clone https://github.com/yourusername/glaucoma-health-economic-model.git
cd glaucoma-health-economic-model
```

## 🚀 Usage

### Quick Start
```python
# Initialize models
ai_model = AIGlaucomaModel()
non_ai_model = NonAIGlaucomaModel()

# Define initial population distribution
initial_dist = {
    'Mild': 0.704,      # 70.4%
    'Moderate': 0.167,   # 16.7%
    'Severe': 0.130,     # 13.0%
    'VI': 0.0            # 0.0%
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

## 📊 Output Analysis

The model generates comprehensive traces including:
- Annual state proportions and transitions
- Total costs and QALYs (discounted and undiscounted)
- State-specific economic outcomes
- Screening program costs
- Uncertainty distributions for all key outcomes

### Key Metrics
1. **Total Costs**: Comparison between AI vs Non-AI strategies
2. **Total QALYs**: Quality-adjusted life years gained
3. **Incremental Cost-Effectiveness Ratio (ICER)**
4. **Probabilistic distributions** for uncertainty analysis


## 🎯 Applications

This model supports:
- **Health Technology Assessment** of AI glaucoma screening programs
- **Resource Allocation** decisions in ophthalmology services
- **Cost-Effectiveness Analysis** for healthcare policymakers
- **Budget Impact Analysis** for implementing AI screening
- **Sensitivity Analysis** of critical model parameters

## ⚠️ Model Limitations

- Assumes perfect adherence to screening and treatment protocols
- Simplified four-state disease progression structure
- Based on specific cost and utility estimates (may vary by healthcare system)
- Does not account for patient heterogeneity beyond modeled disease states
- Limited to 10-year time horizon

## 📈 Validation & Calibration

The model parameters are derived from:
- Published clinical literature on glaucoma progression
- Health economic studies on glaucoma management costs
- Real-world evidence on AI screening performance
- Quality of life studies in glaucoma populations

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Reporting issues
- Suggesting enhancements
- Submitting pull requests
- Code style requirements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this model in your research or policy analysis, please cite:

```bibtex
@software{glaucoma_health_economic_model,
  title={Glaucoma Health Economic Model: AI-Enhanced vs Traditional Screening},
  author={[Federico Felizzi]},
  year={2025},
  url={https://github.com/yourusername/glaucoma-health-economic-model}
}
```

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/glaucoma-health-economic-model/issues)

## 🙏 Acknowledgments

- Clinical advisors and ophthalmology experts
- Health economics research community
- Open-source Python ecosystem contributors

---

**Disclaimer**: This model is for research and educational purposes. Clinical and policy decisions should involve appropriate medical and economic expertise.
