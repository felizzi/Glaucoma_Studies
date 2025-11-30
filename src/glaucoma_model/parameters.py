# parameters.py
"""
Parameter definitions and management for glaucoma health economic model.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class Parameter:
    """Simple parameter class with sampling capability"""
    mean: float
    std: float
    dist_type: str = 'normal'  # normal, gamma, beta

    def sample(self, n=1):
        """Sample from parameter distribution"""
        if self.std == 0:
            return np.full(n, self.mean)
        if self.dist_type == 'gamma':
            shape = (self.mean / self.std) ** 2
            scale = self.std ** 2 / self.mean
            return np.random.gamma(shape, scale, n)
        elif self.dist_type == 'beta':
            alpha = self.mean * (self.mean * (1 - self.mean) / self.std**2 - 1)
            beta = (1 - self.mean) * (self.mean * (1 - self.mean) / self.std**2 - 1)
            return np.random.beta(max(alpha, 0.1), max(beta, 0.1), n)
        else:  # normal
            return np.maximum(0, np.random.normal(self.mean, self.std, n))

print("Parameter class defined!") 

class GlaucomaParameters:
    """All model parameters in one place"""

    def __init__(self, scenario_name="Default",
                 mild_to_moderate_mean=0.15, mild_to_moderate_std=0.05,
                 moderate_to_severe_mean=0.12, moderate_to_severe_std=0.04,
                 severe_to_vi_mean=0.10, severe_to_vi_std=0.03,
                 true_positive_rate=0.90, tp_std=0.05,
                 true_negative_rate=0.85, tn_std=0.05,
                 false_positive_rate=0.15, fp_std=0.05,
                 false_negative_rate=0.10, fn_std=0.05,
                 sensitivity=0.775, sensitivity_std=0.066,
                 specificity=0.954, specificity_std=0.009,
                 # State-specific screening accuracy parameters
                 tp_mild=0.85, tp_mild_std=0.05,
                 tn_mild=0.90, tn_mild_std=0.05,
                 fp_mild=0.10, fp_mild_std=0.05,
                 fn_mild=0.15, fn_mild_std=0.05,
                 tp_moderate=0.90, tp_moderate_std=0.05,
                 tn_moderate=0.88, tn_moderate_std=0.05,
                 fp_moderate=0.12, fp_moderate_std=0.05,
                 fn_moderate=0.10, fn_moderate_std=0.05,
                 tp_severe=0.95, tp_severe_std=0.04,
                 tn_severe=0.85, tn_severe_std=0.05,
                 fp_severe=0.15, fp_severe_std=0.05,
                 fn_severe=0.05, fn_severe_std=0.04,
                 detection_proportion=1.0, detection_std=0.000001,
                 prevalence_general=0.05, prevalence_general_std=0.005,
                 prevalence_dr=0.07, prevalence_dr_std=0.01,
                 screening_cost=10, screening_cost_std=0.01,
                 # Separate screening costs
                 ai_screening_cost=30, ai_screening_cost_std=5,
                 human_screening_cost=75, human_screening_cost_std=15):

        self.scenario_name = scenario_name

        # COSTS (annual, in USD)
        self.costs = {
            'monitoring_mild': Parameter(352, 0.2*352, 'gamma'),
            'monitoring_moderate': Parameter(463, 0.2*463, 'gamma'),
            'monitoring_severe': Parameter(644, 0.2*644, 'gamma'),
            'monitoring_vi': Parameter(576, 0.2*576, 'gamma'),
            'treatment_mild': Parameter(303, 0.2*303, 'gamma'),
            'treatment_moderate': Parameter(429, 0.2*429, 'gamma'),
            'treatment_severe': Parameter(609, 0.2*609, 'gamma'),
            'treatment_vi': Parameter(662, 0.2*662, 'gamma'),
            'other_mild': Parameter(0, 0, 'gamma'),
            'other_moderate': Parameter(0, 0, 'gamma'),
            'other_severe': Parameter(0, 0, 'gamma'),
            'other_vi': Parameter(4186 + 1334, 0.2*(4186 + 1334), 'gamma'),
            'productivity_mild': Parameter(0, 0, 'gamma'),
            'productivity_moderate': Parameter(0, 0, 'gamma'),
            'productivity_severe': Parameter(0, 0, 'gamma'),
            'productivity_vi': Parameter(7630, 0.2*7630, 'gamma'),
            'screening': Parameter(screening_cost, screening_cost_std, 'gamma'),
            'ai_screening': Parameter(ai_screening_cost, ai_screening_cost_std, 'gamma'),
            'human_screening': Parameter(human_screening_cost, human_screening_cost_std, 'gamma'),
        }

        # UTILITIES (0-1 scale)
        self.utilities = {
            'utility_mild': Parameter(0.985, 0.023, 'beta'),
            'utility_moderate': Parameter(0.899, 0.039, 'beta'),
            'utility_severe': Parameter(0.773, 0.046, 'beta'),
            'utility_vi': Parameter(0.634, 0.052, 'beta'),
        }

        # TRANSITION PROBABILITIES
        self.transitions = {
            'mild_to_moderate': Parameter(mild_to_moderate_mean, mild_to_moderate_std, 'beta'),
            'moderate_to_severe': Parameter(moderate_to_severe_mean, moderate_to_severe_std, 'beta'),
            'severe_to_vi': Parameter(severe_to_vi_mean, severe_to_vi_std, 'beta'),
        }

        # SCREENING ACCURACY PARAMETERS (overall)
        self.screening_accuracy = {
            'true_positive_rate': Parameter(true_positive_rate, tp_std, 'beta'),
            'true_negative_rate': Parameter(true_negative_rate, tn_std, 'beta'),
            'false_positive_rate': Parameter(false_positive_rate, fp_std, 'beta'),
            'false_negative_rate': Parameter(false_negative_rate, fn_std, 'beta'),
            'sensitivity': Parameter(sensitivity, sensitivity_std, 'beta'),
            'specificity': Parameter(specificity, specificity_std, 'beta'),
        }

        # STATE-SPECIFIC SCREENING ACCURACY PARAMETERS
        self.screening_accuracy_mild = {
            'tp_mild': Parameter(tp_mild, tp_mild_std, 'beta'),
            'tn_mild': Parameter(tn_mild, tn_mild_std, 'beta'),
            'fp_mild': Parameter(fp_mild, fp_mild_std, 'beta'),
            'fn_mild': Parameter(fn_mild, fn_mild_std, 'beta'),
        }

        self.screening_accuracy_moderate = {
            'tp_moderate': Parameter(tp_moderate, tp_moderate_std, 'beta'),
            'tn_moderate': Parameter(tn_moderate, tn_moderate_std, 'beta'),
            'fp_moderate': Parameter(fp_moderate, fp_moderate_std, 'beta'),
            'fn_moderate': Parameter(fn_moderate, fn_moderate_std, 'beta'),
        }

        self.screening_accuracy_severe = {
            'tp_severe': Parameter(tp_severe, tp_severe_std, 'beta'),
            'tn_severe': Parameter(tn_severe, tn_severe_std, 'beta'),
            'fp_severe': Parameter(fp_severe, fp_severe_std, 'beta'),
            'fn_severe': Parameter(fn_severe, fn_severe_std, 'beta'),
        }

        # DETECTION AND PREVALENCE PARAMETERS
        self.screening_params = {
            'detection_proportion': Parameter(detection_proportion, detection_std, 'beta'),
            'prevalence_general': Parameter(prevalence_general, prevalence_general_std, 'beta'),
            'prevalence_dr': Parameter(prevalence_dr, prevalence_dr_std, 'beta'),
        }

        # DISCOUNT RATES
        self.discount_rates = {
            'cost_discount': Parameter(0.03, 0.01, 'beta'),
            'health_discount': Parameter(0.015, 0.005, 'beta'),
        }

    @classmethod
    def create_ai_pure_scenario(cls, **kwargs):
        """AI PURE SCENARIO - AI transition matrix + AI screening"""
        defaults = {
            'scenario_name': "AI Pure",
            'mild_to_moderate_mean': 0.058, 'mild_to_moderate_std': 0.000303,
            'moderate_to_severe_mean': 0.04, 'moderate_to_severe_std': 0.000253,
            'severe_to_vi_mean': 0.032, 'severe_to_vi_std': 0.00023,
            'true_positive_rate': 0.95, 'tp_std': 0.02,
            'true_negative_rate': 0.92, 'tn_std': 0.02,
            'false_positive_rate': 0.08, 'fp_std': 0.02,
            'false_negative_rate': 0.05, 'fn_std': 0.02,
            'sensitivity' : 0.775, 'sensitivity_std': 0.066,
            'specificity' : 0.954, 'specificity_std': 0.009,
            # AI-enhanced state-specific screening accuracy
            'tp_mild': 0.92, 'tp_mild_std': 0.03,
            'tn_mild': 0.94, 'tn_mild_std': 0.02,
            'fp_mild': 0.06, 'fp_mild_std': 0.02,
            'fn_mild': 0.08, 'fn_mild_std': 0.03,
            'tp_moderate': 0.95, 'tp_moderate_std': 0.02,
            'tn_moderate': 0.93, 'tn_moderate_std': 0.02,
            'fp_moderate': 0.07, 'fp_moderate_std': 0.02,
            'fn_moderate': 0.05, 'fn_moderate_std': 0.02,
            'tp_severe': 0.98, 'tp_severe_std': 0.01,
            'tn_severe': 0.91, 'tn_severe_std': 0.02,
            'fp_severe': 0.09, 'fp_severe_std': 0.02,
            'fn_severe': 0.02, 'fn_severe_std': 0.01,
            'detection_proportion': 0.90, 'detection_std': 0.05,
            # AI screening costs
            'ai_screening_cost': 11.5, 'ai_screening_cost_std': 3, ## AI screening costs include
            'human_screening_cost': 100, 'human_screening_cost_std': 12,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def create_non_ai_pure_scenario(cls, **kwargs):
        """NON-AI PURE SCENARIO - Non-AI transition matrix + Non-AI screening"""
        defaults = {
            'scenario_name': "Non-AI Pure",
            'mild_to_moderate_mean': 0.143, 'mild_to_moderate_std': 0.0323,
            'moderate_to_severe_mean': 0.087, 'moderate_to_severe_std': 0.02603,
            'severe_to_vi_mean': 0.077, 'severe_to_vi_std': 0.02467,
            'true_positive_rate': 0.75, 'tp_std': 0.08,
            'true_negative_rate': 0.80, 'tn_std': 0.08,
            'false_positive_rate': 0.20, 'fp_std': 0.08,
            'false_negative_rate': 0.25, 'fn_std': 0.08,
            # Non-AI state-specific screening accuracy (lower performance)
            'tp_mild': 0.70, 'tp_mild_std': 0.08,
            'tn_mild': 0.85, 'tn_mild_std': 0.06,
            'fp_mild': 0.15, 'fp_mild_std': 0.06,
            'fn_mild': 0.30, 'fn_mild_std': 0.08,
            'tp_moderate': 0.78, 'tp_moderate_std': 0.07,
            'tn_moderate': 0.82, 'tn_moderate_std': 0.06,
            'fp_moderate': 0.18, 'fp_moderate_std': 0.06,
            'fn_moderate': 0.22, 'fn_moderate_std': 0.07,
            'tp_severe': 0.88, 'tp_severe_std': 0.05,
            'tn_severe': 0.78, 'tn_severe_std': 0.07,
            'fp_severe': 0.22, 'fp_severe_std': 0.07,
            'fn_severe': 0.12, 'fn_severe_std': 0.05,
            'detection_proportion': 0.70, 'detection_std': 0.10,
            # Human screening costs (higher due to specialist time)
            'ai_screening_cost': 0.01, 'ai_screening_cost_std': 0.0001,
            'human_screening_cost': 0.01, 'human_screening_cost_std': 0.0000001,
        }
        defaults.update(kwargs)
        instance = cls(**defaults)

        instance._set_non_ai_cost_structure()
        return instance

    def _set_non_ai_cost_structure(self):
        """Set Non-AI cost structure: ONLY VI patients incur costs"""
        zero_cost = Parameter(0, 0, 'gamma')

        # Zero out all costs except VI
        self.costs.update({
            # NO costs for undetected cases
            'monitoring_mild': zero_cost,
            'monitoring_moderate': zero_cost,
            'monitoring_severe': zero_cost,
            'treatment_mild': zero_cost,
            'treatment_moderate': zero_cost,
            'treatment_severe': zero_cost,
            'other_mild': zero_cost,
            'other_moderate': zero_cost,
            'other_severe': zero_cost,
            'productivity_mild': zero_cost,
            'productivity_moderate': zero_cost,
            'productivity_severe': zero_cost,

            # VI costs remain the same (clinically obvious)
            # 'monitoring_vi': unchanged
            # 'treatment_vi': unchanged
            # 'other_vi': unchanged
            # 'productivity_vi': unchanged
        })

        print(f"Applied Non-AI cost structure: Only VI patients incur costs")

    def sample_all(self):
        """Sample all parameters once"""
        sample = {}
        for category in [self.costs, self.utilities, self.transitions,
                        self.screening_accuracy, self.screening_accuracy_mild,
                        self.screening_accuracy_moderate, self.screening_accuracy_severe,
                        self.screening_params, self.discount_rates]:
            for name, param in category.items():
                sample[name] = param.sample(1)[0]
        return sample

    def get_summary(self):
        """Get parameter summary as DataFrame"""
        data = []
        for category_name, category in [('Costs', self.costs),
                                       ('Utilities', self.utilities),
                                       ('Transitions', self.transitions),
                                       ('Screening_Accuracy', self.screening_accuracy),
                                       ('Screening_Accuracy_Mild', self.screening_accuracy_mild),
                                       ('Screening_Accuracy_Moderate', self.screening_accuracy_moderate),
                                       ('Screening_Accuracy_Severe', self.screening_accuracy_severe),
                                       ('Screening_Params', self.screening_params),
                                       ('Discount_Rates', self.discount_rates)]:
            for name, param in category.items():
                data.append({
                    'Category': category_name,
                    'Parameter': name,
                    'Mean': param.mean,
                    'Std': param.std,
                    'Distribution': param.dist_type
                })
        return pd.DataFrame(data)

    def get_screening_cost(self, screening_type='combined'):
        """Get screening cost based on type"""
        if screening_type == 'ai_only':
            return self.costs['ai_screening']
        elif screening_type == 'human_only':
            return self.costs['human_screening']
        elif screening_type == 'combined':
            return self.costs['screening']
        else:
            raise ValueError(f"Unknown screening type: {screening_type}. Use 'ai_only', 'human_only', or 'combined'")

    def get_state_specific_accuracy(self, state):
        """Get screening accuracy parameters for a specific state"""
        if state == 'mild':
            return self.screening_accuracy_mild
        elif state == 'moderate':
            return self.screening_accuracy_moderate
        elif state == 'severe':
            return self.screening_accuracy_severe
        else:
            raise ValueError(f"Unknown state: {state}. Use 'mild', 'moderate', or 'severe'")

print("Enhanced GlaucomaParameters class defined with state-specific screening accuracy!")