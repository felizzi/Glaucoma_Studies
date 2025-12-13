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

## 📂 Repository Structure

```
glaucoma-health-economic-model/
│
├── src/                          # Source code directory
│   └── data/                     # Data files and datasets
│
├── glaucoma_model/              # Core model package
│   ├── __init__.py              # Package initialization
│   ├── analysis.py              # Analysis functions and utilities
│   ├── model.py                 # Core Markov model implementation
│   ├── parameters.py            # Model parameters (costs, utilities, transitions)
│   └── utils.py                 # Helper functions and utilities
│
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── 01_model_setup_restructured.ipynb
│   ├── 02_prevalence_sensitivity_rerun.ipynb
│   ├── 03_productivity_sensitivity_analysis.ipynb
│   ├── 04_starting_age_sensitivity_analysis.ipynb
│   ├── 05_ai_cost_sensitivity_analysis.ipynb
│   ├── 06_fp_disutility_sensitivity_analysis.ipynb
│   ├── 07_screening_frequency_sensitivity_analysis.ipynb
│   ├── 08_max_ai_cost_by_prevalence_optimized.ipynb
│   └── Mortality_loader.ipynb
│
├── _old_/                       # Archived files from previous versions
├── README.md                     # This file
└── LICENSE                       # MIT License
```

## 📓 Notebooks Directory

The analysis is organized into a series of Jupyter notebooks that build upon each other. Each notebook is self-contained but follows a logical progression through the model development and sensitivity analyses.

### Core Model Setup

#### 01_model_setup_restructured.ipynb
**Purpose**: Foundation of the entire analysis  
**Key Components**:
- Markov model structure definition (4 health states)
- Transition probability matrices for AI and non-AI strategies
- Cost parameters (screening, monitoring, treatment, productivity)
- Health utility values and QALY calculations
- Initial population distribution setup
- Deterministic and probabilistic sensitivity analysis framework

**Outputs**:
- Base case results for both strategies
- Cost-effectiveness comparison
- Distribution plots from PSA
- State transition diagrams

**When to Use**: Start here to understand the model fundamentals and run base case analyses

---

### Sensitivity Analyses

#### 02_prevalence_sensitivity_rerun.ipynb
**Purpose**: Examine how glaucoma prevalence affects cost-effectiveness  
**Analysis Type**: One-way sensitivity analysis  
**Parameter Varied**: Initial glaucoma prevalence (range: 1% to 10%)  
**Key Insights**:
- Impact of prevalence on screening program value
- Threshold prevalence for cost-effectiveness
- Population-specific screening strategies

**Outputs**:
- Tornado diagrams
- Prevalence vs ICER curves
- Break-even prevalence analysis

---

#### 03_productivity_sensitivity_analysis.ipynb
**Purpose**: Evaluate productivity loss impacts across disease states  
**Analysis Type**: Scenario analysis with productivity costs  
**Parameters Analyzed**:
- Productivity losses by disease severity
- Societal vs healthcare perspective comparisons
- Age-dependent productivity valuations

**Key Insights**:
- Societal value of preventing vision loss
- Economic burden beyond direct medical costs
- Return on investment from employer perspective

**Outputs**:
- Productivity-adjusted cost-effectiveness ratios
- Comparative perspective analysis (healthcare vs societal)
- Waterfall charts of cost components

---

#### 04_starting_age_sensitivity_analysis.ipynb
**Purpose**: Determine optimal screening initiation age  
**Analysis Type**: Age-stratified analysis  
**Age Ranges Tested**: 40, 50, 60, 70, 80 years  
**Key Considerations**:
- Age-specific glaucoma prevalence
- Remaining quality-adjusted life years
- Cost-effectiveness by age cohort
- Life expectancy adjustments

**Outputs**:
- Age-specific ICERs
- Optimal screening age recommendations
- Age-adjusted cost per QALY gained

---

#### 05_ai_cost_sensitivity_analysis.ipynb
**Purpose**: Test robustness to AI screening cost variations  
**Analysis Type**: One-way sensitivity analysis  
**Cost Range**: $5 to $150 per screening  
**Key Questions**:
- Maximum acceptable AI screening cost
- Cost threshold for cost-effectiveness
- Pricing strategies for AI implementation

**Outputs**:
- Cost threshold analysis
- Break-even AI screening price
- Value-based pricing recommendations

---

#### 06_fp_disutility_sensitivity_analysis.ipynb
**Purpose**: Analyze false positive rates and quality of life impacts  
**Analysis Type**: Multi-parameter sensitivity analysis  
**Parameters Examined**:
- False positive rates (specificity variations)
- Anxiety and disutility from false positives
- Follow-up burden and costs

**Key Insights**:
- Optimal balance between sensitivity and specificity
- Patient experience considerations
- Trade-offs in screening accuracy

**Outputs**:
- Receiver operating characteristic (ROC) analysis
- Disutility-adjusted QALYs
- Screening performance optimization

---

#### 07_screening_frequency_sensitivity_analysis.ipynb
**Purpose**: Compare different screening interval strategies  
**Screening Frequencies Tested**:
- Annual screening
- Biennial screening
- Every 3 years
- Every 5 years
- Risk-based variable screening

**Key Insights**:
- Optimal screening frequency by risk level
- Diminishing returns from frequent screening
- Resource allocation optimization

**Outputs**:
- Frequency vs effectiveness curves
- Cost per additional case detected
- Recommended screening intervals by population

---

#### 08_max_ai_cost_by_prevalence_optimized.ipynb
**Purpose**: Advanced optimization analysis  
**Analysis Type**: Two-way sensitivity analysis  
**Parameters**: AI cost × prevalence interaction  
**Methodology**:
- Grid search optimization
- Cost-effectiveness acceptability curves (CEAC)
- Net monetary benefit calculations

**Key Insights**:
- Maximum willingness to pay for AI screening by prevalence
- Population targeting strategies
- Resource allocation across different prevalence settings

**Outputs**:
- 2D heat maps of cost-effectiveness
- Optimal AI pricing by prevalence
- Decision rules for implementation

---

#### Mortality_loader.ipynb
**Purpose**: Load and process mortality data for model calibration  
**Data Sources**:
- Life tables by age and gender
- Glaucoma-specific mortality adjustments
- Comorbidity impact on survival

**Functions**:
- Data import and cleaning
- Life table integration
- Survival probability calculations

**When to Use**: For model calibration and validation; typically run before main analyses

---

### Recommended Workflow

#### For New Users:
1. **Start**: `01_model_setup_restructured.ipynb` - Understand the base model
2. **Explore**: Run sensitivity analyses (02-07) to understand parameter uncertainty
3. **Advanced**: `08_max_ai_cost_by_prevalence_optimized.ipynb` for optimization

#### For Researchers:
1. Review `Mortality_loader.ipynb` for data sources
2. Examine `01_model_setup_restructured.ipynb` for model structure
3. Replicate or modify sensitivity analyses as needed
4. Use `08_max_ai_cost_by_prevalence_optimized.ipynb` for policy recommendations

#### For Policymakers:
1. Base case results: `01_model_setup_restructured.ipynb`
2. Key sensitivity analyses: `02, 04, 05` for decision-relevant parameters
3. Implementation guidance: `07, 08` for program design

---

## 🏥 Model Structure

### Disease States
- **Mild Glaucoma** - Early stage with minimal vision loss
- **Moderate Glaucoma** - Progressive vision field defects
- **Severe Glaucoma** - Significant vision impairment
- **Visual Impairment (VI)** - Absorbing state with substantial vision loss

### Transition Probabilities
The model incorporates different progression rates based on screening strategy:

| Transition | AI Strategy | Non-AI Strategy | Relative Risk Reduction |
|------------|-------------|-----------------|------------------------|
| Mild → Moderate | 0.058 | 0.143 | 59.4% |
| Moderate → Severe | 0.040 | 0.087 | 54.0% |
| Severe → VI | 0.032 | 0.077 | 58.4% |

**Rationale**: AI-enhanced screening enables earlier detection and intervention, slowing disease progression through:
- Earlier initiation of IOP-lowering treatment
- More intensive monitoring and treatment adjustment
- Better patient adherence through improved care coordination

---

## 💰 Economic Parameters

### Annual Costs (USD/EUR)
| State | Monitoring | Treatment | Productivity Loss | Total (AI Strategy) |
|-------|------------|-----------|-------------------|---------------------|
| Mild | €352 | €303 | €0 | €655 |
| Moderate | €463 | €429 | €0 | €892 |
| Severe | €644 | €609 | €0 | €1,253 |
| VI | €576 | €662 | €7,630 | €8,868 |

### Screening Costs
- **AI Screening**: $11.50 per patient
  - Includes: Automated imaging, AI analysis, ophthalmologist review
- **Traditional Screening**: $100.00 per patient
  - Includes: Comprehensive eye exam, tonometry, visual field testing

### Health Utilities (0-1 scale)
Quality of life weights based on validated instruments (EQ-5D, NEI-VFQ-25):

| State | Mean Utility | Standard Deviation | 95% CI |
|-------|--------------|-------------------|---------|
| Mild | 0.985 | 0.023 | [0.940, 1.000] |
| Moderate | 0.899 | 0.039 | [0.823, 0.975] |
| Severe | 0.773 | 0.046 | [0.683, 0.863] |
| VI | 0.634 | 0.052 | [0.532, 0.736] |

### Screening Performance Characteristics

#### AI Strategy
- **Sensitivity**: 77.5% (95% CI: 73.2% - 81.8%)
- **Specificity**: 95.4% (95% CI: 94.1% - 96.7%)
- **Positive Predictive Value**: Variable by prevalence
- **Negative Predictive Value**: Variable by prevalence

#### Traditional Strategy
- **Sensitivity**: 65.0% (95% CI: 60.0% - 70.0%)
- **Specificity**: 90.0% (95% CI: 87.5% - 92.5%)

**Source**: Derived from meta-analyses of AI diagnostic accuracy studies and real-world ophthalmology practice

---

## 🛠️ Installation

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scipy jupyter
```

### Optional Dependencies (for advanced analyses)
```bash
pip install plotly ipywidgets tqdm scikit-learn
```

### Clone Repository
```bash
git clone https://github.com/yourusername/glaucoma-health-economic-model.git
cd glaucoma-health-economic-model
```

### Set Up Python Environment
```bash
# Using conda
conda create -n glaucoma-model python=3.9
conda activate glaucoma-model
pip install -r requirements.txt

# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Usage

### Quick Start
```python
from glaucoma_model.model import AIGlaucomaModel, NonAIGlaucomaModel
from glaucoma_model.parameters import DEFAULT_PARAMETERS

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

# Calculate incremental cost-effectiveness
incremental_cost = ai_results['total_cost'] - non_ai_results['total_cost']
incremental_qaly = ai_results['total_qaly'] - non_ai_results['total_qaly']
icer = incremental_cost / incremental_qaly

print(f"Incremental Cost: ${incremental_cost:,.2f}")
print(f"Incremental QALYs: {incremental_qaly:.4f}")
print(f"ICER: ${icer:,.2f} per QALY gained")
```

### Probabilistic Sensitivity Analysis
```python
# Run probabilistic analysis (5,000 iterations)
ai_psa_results = ai_model.run_probabilistic(
    n_iterations=5000, 
    initial_dist=initial_dist,
    seed=42  # For reproducibility
)
non_ai_psa_results = non_ai_model.run_probabilistic(
    n_iterations=5000, 
    initial_dist=initial_dist,
    seed=42
)

# Calculate cost-effectiveness acceptability
from glaucoma_model.analysis import calculate_ceac

ceac = calculate_ceac(
    ai_psa_results, 
    non_ai_psa_results,
    wtp_thresholds=range(0, 100000, 1000)
)
```

### Custom Parameter Scenario
```python
# Modify parameters for a specific scenario
custom_params = DEFAULT_PARAMETERS.copy()
custom_params['ai_screening_cost'] = 20.0  # $20 per AI screening
custom_params['prevalence'] = 0.05  # 5% prevalence

# Create model with custom parameters
ai_model_custom = AIGlaucomaModel(parameters=custom_params)
results = ai_model_custom.run_deterministic(initial_dist=initial_dist)
```

---

## 📊 Output Analysis

### Generated Outputs

#### Deterministic Results
The model generates comprehensive traces including:
- **State Distributions**: Annual proportions in each health state
- **Transition Flows**: Number of patients transitioning between states
- **Cost Breakdown**: 
  - Screening costs
  - Monitoring costs
  - Treatment costs
  - Productivity losses
- **QALY Accumulation**: Discounted and undiscounted quality-adjusted life years
- **Incremental Analyses**: Cost-effectiveness ratios and net monetary benefit

#### Probabilistic Results
- **Distribution Plots**: Histograms and density plots for all outcomes
- **Scatter Plots**: Cost-effectiveness planes
- **CEAC**: Cost-effectiveness acceptability curves
- **Confidence Intervals**: 95% credible intervals for all estimates
- **Sensitivity Rankings**: Parameters with greatest impact on results

### Key Metrics

1. **Total Costs**: Comparison between AI vs Non-AI strategies
   - Healthcare sector costs
   - Societal costs (including productivity)
   
2. **Total QALYs**: Quality-adjusted life years gained
   - Per 1,000 patients screened
   - Discounted at 3% annually

3. **Incremental Cost-Effectiveness Ratio (ICER)**
   - Incremental cost per QALY gained
   - Compared to willingness-to-pay thresholds ($50K-$150K/QALY)

4. **Net Monetary Benefit (NMB)**
   - Value-based metric at different WTP thresholds
   - Probability of cost-effectiveness

5. **Budget Impact**
   - Total program costs over 10 years
   - Per-patient screening program costs
   - Return on investment timelines

---

## 🎯 Applications

This model supports:

### Health Technology Assessment (HTA)
- Evaluate AI screening technologies for reimbursement decisions
- Determine value-based pricing for AI diagnostic tools
- Inform coverage decisions by payers and health systems

### Resource Allocation
- Optimize allocation of ophthalmology resources
- Determine screening frequency and target populations
- Balance early detection benefits vs screening costs

### Healthcare Policy
- Design population-level glaucoma screening programs
- Set evidence-based screening guidelines
- Evaluate different implementation strategies

### Budget Planning
- Estimate financial impact of implementing AI screening
- Project long-term cost savings from disease prevention
- Plan capacity needs for eye care services

### Research Prioritization
- Identify high-value areas for further research
- Quantify value of information for key uncertainties
- Guide clinical trial design and outcome selection

---

## ⚠️ Model Limitations

Users should be aware of the following limitations:

### Structural Assumptions
- **Four-state simplification**: Real glaucoma progression is more continuous
- **Homogeneous populations**: Does not account for individual risk factors beyond modeled states
- **No regression**: Assumes disease can only progress or remain stable
- **Single intervention comparison**: Does not model combination strategies or sequential testing

### Time Horizon
- **10-year horizon**: May not capture lifetime benefits and costs
- **Annual cycles**: Approximates continuous progression with discrete time steps
- **No mortality modeling**: Assumes equal mortality across strategies (conservative)

### Clinical Parameters
- **Perfect adherence assumption**: Model assumes 100% adherence to screening and treatment
- **Fixed screening intervals**: Does not model risk-based variable screening
- **Treatment effectiveness**: Assumed constant across all disease stages

### Economic Parameters
- **Cost estimates**: Based on specific healthcare system (may vary by country/region)
- **Utility values**: Derived from generic instruments, may not capture all glaucoma-specific impacts
- **No indirect costs beyond productivity**: Doesn't include caregiver burden, transportation, etc.

### Data Quality
- **Parameter uncertainty**: Some parameters based on limited evidence or expert opinion
- **Generalizability**: Results most applicable to populations similar to those in source studies
- **External validity**: Model calibration assumes specific population characteristics

### Implementation Considerations
- **Not modeled**: Training needs, workflow integration, technology infrastructure
- **Organizational factors**: Assumes sufficient capacity and expertise for AI implementation
- **Equity considerations**: Does not address disparities in access or outcomes

**Recommendation**: Conduct local validation and calibration before using results for decision-making in specific contexts.

---

## 📈 Validation & Calibration

### Model Validation Approach

#### Face Validity
- Clinical expert review of model structure
- Verification of disease progression pathways
- Confirmation of cost and utility parameters

#### Internal Validity
- Extreme value testing (0% and 100% parameter values)
- Logical consistency checks
- Transition probability sum validation

#### External Validity
- Comparison with published cost-effectiveness studies
- Cross-validation against observational cohort data
- Reproduction of known epidemiological patterns

### Data Sources

#### Clinical Parameters
- **Transition Probabilities**: 
  - Collaborative Initial Glaucoma Treatment Study (CIGTS)
  - Early Manifest Glaucoma Trial (EMGT)
  - Meta-analyses of glaucoma progression studies

#### Economic Parameters
- **Costs**: 
  - National health insurance claims databases
  - Published microcosting studies
  - Expert panel estimates for novel interventions

#### Health Utilities
- **Quality of Life Weights**:
  - EQ-5D population surveys
  - National Eye Institute Visual Function Questionnaire (NEI-VFQ-25)
  - Glaucoma-specific utility studies

#### AI Performance
- **Diagnostic Accuracy**:
  - Systematic reviews of AI diagnostic studies
  - Real-world evidence from pilot implementations
  - Comparative effectiveness studies vs standard care

### Calibration Targets
The model was calibrated to match:
- Population-level glaucoma prevalence
- Age-specific incidence rates
- Distribution of disease severity at diagnosis
- Known progression rates from landmark trials

---

## 🤝 Contributing

We welcome contributions from researchers, clinicians, health economists, and developers!

### Ways to Contribute

#### Report Issues
- Bug reports in model implementation
- Documentation improvements
- Parameter updates with new evidence

#### Suggest Enhancements
- New sensitivity analyses
- Alternative model structures
- Additional outcome measures
- Visualization improvements

#### Submit Code
- New features or analyses
- Performance optimizations
- Testing improvements
- Documentation updates

### Contribution Guidelines

1. **Fork the repository** and create a feature branch
2. **Follow PEP 8** style guidelines for Python code
3. **Add tests** for new functionality
4. **Update documentation** including docstrings and README
5. **Submit a pull request** with clear description of changes

### Code Style Requirements
```python
# Use descriptive variable names
transition_probability = 0.058  # Good
tp = 0.058  # Avoid

# Add docstrings to functions
def calculate_qaly(utility: float, duration: float) -> float:
    """
    Calculate quality-adjusted life years.
    
    Parameters:
        utility (float): Health utility value (0-1 scale)
        duration (float): Time duration in years
        
    Returns:
        float: Quality-adjusted life years
    """
    return utility * duration

# Use type hints
def run_model(n_iterations: int, seed: int = None) -> dict:
    pass
```

### Pull Request Process
1. Ensure all tests pass
2. Update the README with details of changes if applicable
3. Increase version numbers following semantic versioning
4. PR will be merged once you have approval from maintainers

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ Commercial use permitted
- ✅ Modification permitted
- ✅ Distribution permitted
- ✅ Private use permitted
- ⚠️ No liability or warranty provided

---

## 📚 Citation

If you use this model in your research, policy analysis, or health technology assessment, please cite:

```bibtex
@software{glaucoma_health_economic_model,
  title={Glaucoma Health Economic Model: AI-Enhanced vs Traditional Screening},
  author={Felizzi, Federico},
  year={2025},
  url={https://github.com/yourusername/glaucoma-health-economic-model},
  version={1.0.0},
  note={A Markov model for evaluating cost-effectiveness of AI-enhanced glaucoma screening}
}
```

### Related Publications
If you publish work using this model, please:
1. Cite the software using the BibTeX above
2. Reference key data sources used in parameterization
3. Note any modifications made to the base model

---

## 📞 Contact

### Support Channels

**GitHub Issues**: [Create an issue](https://github.com/yourusername/glaucoma-health-economic-model/issues)
- For bug reports, feature requests, and technical questions

**Email**: [Your email]
- For collaboration inquiries and complex questions

**Discussions**: [GitHub Discussions](https://github.com/yourusername/glaucoma-health-economic-model/discussions)
- For general questions and community discussions

### Collaboration Opportunities
We are interested in:
- Multi-country validation studies
- Extensions to other ophthalmologic conditions
- Real-world implementation studies
- Methodological improvements

---

## 🙏 Acknowledgments

### Contributors
- **Clinical Advisors**: Ophthalmology experts who provided clinical insights
- **Health Economics Reviewers**: Methodological guidance and validation
- **Data Providers**: Organizations that shared cost and outcomes data

### Funding
[If applicable, acknowledge funding sources]

### Open Source Community
This project builds upon the excellent work of:
- **NumPy/SciPy**: Scientific computing infrastructure
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Visualization capabilities
- **Jupyter**: Interactive computing environment

### Inspiration
Model structure inspired by published health economic models in ophthalmology and guidance from:
- ISPOR Good Research Practices
- CHEERS 2022 Reporting Guidelines
- NICE Methods Guide

---

## 📖 Additional Resources

### Further Reading
- **ISPOR**: [Cost-Effectiveness Analysis Guidelines](https://www.ispor.org)
- **NICE**: [Guide to Methods of Technology Appraisal](https://www.nice.org.uk/process/pmg9)
- **WHO**: [Global Report on Vision](https://www.who.int/publications/i/item/9789241516570)

### Related Models
- **Diabetic Retinopathy Screening Models**: Similar AI implementation considerations
- **Cataract Surgery Economic Models**: Ophthalmology cost-effectiveness examples
- **AMD Treatment Models**: Chronic eye disease management parallels

### Training Materials
- [Health Economic Modeling in R](https://r-hta.org/)
- [Decision Modeling for Health Economic Evaluation](https://www.herc.ox.ac.uk/downloads/decision-modelling-for-health-economic-evaluation)
- [ISPOR Short Courses](https://www.ispor.org/conferences-education/education-training/short-courses)

---

## 📋 Version History

### Version 1.0.0 (Current)
- Initial release with base model and 8 sensitivity analyses
- Probabilistic sensitivity analysis with 5,000 iterations
- Comprehensive documentation and example notebooks

### Planned Updates
- **Version 1.1**: Add lifetime horizon option
- **Version 1.2**: Incorporate real-world evidence from pilot studies
- **Version 2.0**: Multi-country adaptation with region-specific parameters

---

## ⚖️ Disclaimer

**Important**: This model is intended for research, educational, and health technology assessment purposes only. 

- **Not Clinical Guidance**: Results should not be used as sole basis for individual patient care decisions
- **Professional Judgment Required**: Clinical and policy decisions should involve appropriate medical and economic expertise
- **Local Validation Needed**: Parameters should be validated for local context before use
- **No Warranties**: The model is provided "as-is" without warranty of any kind
- **User Responsibility**: Users are responsible for verifying results and ensuring appropriate application

By using this model, you acknowledge these limitations and agree to use it responsibly in accordance with professional standards and ethical guidelines.

---

**Last Updated**: December 2025  
**Model Version**: 1.0.0  
**Documentation Version**: 1.0.0