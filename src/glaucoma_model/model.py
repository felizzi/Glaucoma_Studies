"""
Core model classes for glaucoma health economic evaluation.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from .parameters import GlaucomaParameters

class BaseGlaucomaModel:
    """Base class for Glaucoma Health Economic Models"""

    def __init__(self, params=None, starting_age=60, mortality_table=None):
        self.params = params or GlaucomaParameters()
        self.states = ['Mild', 'Moderate', 'Severe', 'VI', 'Dead']
        self.scenario_name = self.params.scenario_name
        self.starting_age = starting_age
        self.mortality_table = mortality_table or self._get_default_mortality_table()

    def _get_default_mortality_table(self):
        """
        Default mortality table with q(x) values
        This should be replaced with actual life table data
        Based on typical developed country life tables
        """
        return {
            40: 0.00143, 41: 0.00154, 42: 0.00166, 43: 0.00179, 44: 0.00194,
            45: 0.00210, 46: 0.00227, 47: 0.00246, 48: 0.00268, 49: 0.00292,
            50: 0.00319, 51: 0.00349, 52: 0.00382, 53: 0.00419, 54: 0.00460,
            55: 0.00505, 56: 0.00555, 57: 0.00610, 58: 0.00671, 59: 0.00738,
            60: 0.00812, 61: 0.00894, 62: 0.00983, 63: 0.01082, 64: 0.01190,
            65: 0.01309, 66: 0.01439, 67: 0.01581, 68: 0.01737, 69: 0.01907,
            70: 0.02094, 71: 0.02298, 72: 0.02522, 73: 0.02767, 74: 0.03035,
            75: 0.03329, 76: 0.03650, 77: 0.04002, 78: 0.04387, 79: 0.04808,
            80: 0.05270, 81: 0.05776, 82: 0.06331, 83: 0.06940, 84: 0.07607,
            85: 0.08339, 86: 0.09141, 87: 0.10020, 88: 0.10982, 89: 0.12035,
            90: 0.13187, 91: 0.14446, 92: 0.15821, 93: 0.17321, 94: 0.18956,
            95: 0.20736, 96: 0.22672, 97: 0.24775, 98: 0.27057, 99: 0.29531,
            100: 0.32210, 101: 0.35109, 102: 0.38243, 103: 0.41628, 104: 0.45281,
            105: 0.49220, 106: 0.53464, 107: 0.58032, 108: 0.62946, 109: 0.68227,
            110: 1.00000  # Assume death at 110
        }

    def load_mortality_table_from_file(self, filepath_male, filepath_female, male_proportion=0.5, age_col='Age', qx_col='qx'):
        """
        Load mortality table from CSV file
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file containing mortality table
        age_col : str
            Name of the age column (default: 'Age')
        qx_col : str
            Name of the q(x) column (default: 'qx')
        
        Returns:
        --------
        dict : Dictionary mapping age to mortality probability
        """
        import pandas as pd
        df_male = pd.read_csv(filepath_male)
        df_female = pd.read_csv(filepath_female)
        df = pd.DataFrame()
        df[age_col] = df_male[age_col]
        df[qx_col] = (male_proportion * df_male [qx_col]) + ((1 - male_proportion) * df_female[qx_col])  
        mortality_dict = dict(zip(df[age_col], df[qx_col]))
        self.mortality_table = mortality_dict
        return mortality_dict

    def get_age_specific_mortality(self, age, sample):
        """
        Get age-specific mortality rate q(x) from mortality table
        
        Parameters:
        -----------
        age : float or int
            Current age
        sample : dict
            Dictionary of sampled parameters (for uncertainty around mortality)
        
        Returns:
        --------
        float : Probability of dying in the next year
        """
        age_int = int(round(age))
        
        # Get base mortality from table
        if age_int in self.mortality_table:
            base_qx = self.mortality_table[age_int]
        else:
            # If age not in table, use nearest available or extrapolate
            if age_int < min(self.mortality_table.keys()):
                base_qx = self.mortality_table[min(self.mortality_table.keys())]
            else:
                base_qx = self.mortality_table[max(self.mortality_table.keys())]
        
        # Optional: Apply uncertainty/variation to mortality rates in PSA
        # mortality_adjustment = sample.get('mortality_adjustment_factor', 1.0)
        # base_qx = base_qx * mortality_adjustment
        
        return np.clip(base_qx, 0, 1)

    def get_state_specific_mortality_multiplier(self, state, sample):
        """
        Get mortality multiplier for each health state (hazard ratio vs general population)
        These represent excess mortality risk due to glaucoma severity
        """
        multipliers = {
            'Mild': sample.get('mortality_multiplier_mild', 1.0), ## modify the mortality multipliers as needed
            'Moderate': sample.get('mortality_multiplier_moderate', 1.05),
            'Severe': sample.get('mortality_multiplier_severe', 1.10),
            'VI': sample.get('mortality_multiplier_vi', 1.20),
            'Dead': 0.0
        }
        return multipliers.get(state, 1.0)

    def get_transition_matrix(self, sample, age):
        """Build transition matrix from sampled parameters including age-specific mortality"""
        p1 = sample['mild_to_moderate']
        p2 = sample['moderate_to_severe']
        p3 = sample['severe_to_vi']
        p1, p2, p3 = np.clip([p1, p2, p3], 0, 1)
        
        # Get age-specific base mortality q(x) from mortality table
        base_qx = self.get_age_specific_mortality(age, sample)
        
        # Apply state-specific mortality multipliers (hazard ratios)
        mort_mild = base_qx * self.get_state_specific_mortality_multiplier('Mild', sample)
        mort_moderate = base_qx * self.get_state_specific_mortality_multiplier('Moderate', sample)
        mort_severe = base_qx * self.get_state_specific_mortality_multiplier('Severe', sample)
        mort_vi = base_qx * self.get_state_specific_mortality_multiplier('VI', sample)
        
        # Clip mortality rates to valid range [0, 1]
        mort_mild, mort_moderate, mort_severe, mort_vi = np.clip(
            [mort_mild, mort_moderate, mort_severe, mort_vi], 0, 1
        )
        
        # Ensure transition probabilities + mortality don't exceed 1
        p1 = np.clip(p1, 0, 1 - mort_mild)
        p2 = np.clip(p2, 0, 1 - mort_moderate)
        p3 = np.clip(p3, 0, 1 - mort_severe)
        
        # Build transition matrix with age-specific mortality
        # Each row sums to 1
        return np.array([
            # From Mild: stay, progress to Moderate, skip, skip, die
            [1 - p1 - mort_mild, p1, 0, 0, mort_mild],
            # From Moderate: skip, stay, progress to Severe, skip, die
            [0, 1 - p2 - mort_moderate, p2, 0, mort_moderate],
            # From Severe: skip, skip, stay, progress to VI, die
            [0, 0, 1 - p3 - mort_severe, p3, mort_severe],
            # From VI: skip, skip, skip, stay, die
            [0, 0, 0, 1 - mort_vi, mort_vi],
            # From Dead: stay dead
            [0, 0, 0, 0, 1]
        ])

    def simulate_cohort(self, initial_dist, years, sample):
        """Simulate cohort over time with age-dependent mortality"""
        n_states = len(self.states)
        cohort = np.zeros((years + 1, n_states))
        cohort[0] = initial_dist
        
        # Store age-specific transition matrices and ages for tracing
        transition_matrices = []
        ages = []
        
        for year in range(years):
            current_age = self.starting_age + year
            ages.append(current_age)
            trans_matrix = self.get_transition_matrix(sample, current_age)
            transition_matrices.append(trans_matrix)
            cohort[year + 1] = cohort[year] @ trans_matrix
        
        return cohort, transition_matrices, ages

    def create_detailed_traces(self, cohort, costs, qalys, costs_disc, qalys_disc,
                              state_costs, state_utilities, sample, screening_costs,
                              transition_matrices=None, ages=None):
        """Create detailed year-by-year traces"""
        years = len(cohort)
        trace_data = []

        for year in range(years):
            current_age = ages[year] if ages and year < len(ages) else self.starting_age + year
            cost_discount_factor = 1 / (1 + sample['cost_discount']) ** year
            health_discount_factor = 1 / (1 + sample['health_discount']) ** year
            state_costs_year = cohort[year][:4] * state_costs  # Only living states have costs
            state_utilities_year = cohort[year][:4] * state_utilities  # Only living states have utilities

            row = {
                'Year': year,
                'Age': current_age,
                'Prop_Mild': cohort[year][0],
                'Prop_Moderate': cohort[year][1],
                'Prop_Severe': cohort[year][2],
                'Prop_VI': cohort[year][3],
                'Prop_Dead': cohort[year][4],
                'Prop_Alive': 1 - cohort[year][4],
                'Total_Cost': costs[year],
                'Total_QALY': qalys[year],
                'Total_Cost_Disc': costs_disc[year],
                'Total_QALY_Disc': qalys_disc[year],
                'Screening_Cost': screening_costs[year],
                'Cost_Mild': state_costs_year[0],
                'Cost_Moderate': state_costs_year[1],
                'Cost_Severe': state_costs_year[2],
                'Cost_VI': state_costs_year[3],
                'QALY_Mild': state_utilities_year[0],
                'QALY_Moderate': state_utilities_year[1],
                'QALY_Severe': state_utilities_year[2],
                'QALY_VI': state_utilities_year[3],
                'Cost_Discount_Factor': cost_discount_factor,
                'Health_Discount_Factor': health_discount_factor,
            }
            
            # Add age-specific mortality rates if transition matrices available
            if transition_matrices and year < len(transition_matrices):
                row['Mortality_Rate_Mild'] = transition_matrices[year][0, 4]
                row['Mortality_Rate_Moderate'] = transition_matrices[year][1, 4]
                row['Mortality_Rate_Severe'] = transition_matrices[year][2, 4]
                row['Mortality_Rate_VI'] = transition_matrices[year][3, 4]
                
                # Also store base mortality from table
                base_mort = self.get_age_specific_mortality(current_age, sample)
                row['Base_Mortality_qx'] = base_mort
            
            trace_data.append(row)

        return pd.DataFrame(trace_data)

    # Abstract methods to be implemented by subclasses
    def calculate_outcomes(self, cohort, sample, include_screening=True, population_type='general', 
                          transition_matrices=None, ages=None):
        raise NotImplementedError("Subclasses must implement calculate_outcomes")

    def run_deterministic(self, initial_dist=None, years=10, include_screening=True, 
                         population_type='general', starting_age=None):
        raise NotImplementedError("Subclasses must implement run_deterministic")

    def run_probabilistic(self, n_iterations=1000, initial_dist=None, years=10,
                         include_screening=False, population_type='general', random_seed=42, 
                         return_traces=False, starting_age=None):
        raise NotImplementedError("Subclasses must implement run_probabilistic")


class AIGlaucomaModel(BaseGlaucomaModel):
    """AI-Enhanced Glaucoma Model with advanced screening and early detection"""

    def __init__(self, params=None, starting_age=60, mortality_table=None):
        if params is None:
            params = GlaucomaParameters.create_ai_pure_scenario()
        super().__init__(params, starting_age, mortality_table)
        self.model_type = "AI_Enhanced"

    def calculate_outcomes(self, cohort, sample, include_screening=True, population_type='general',
                          transition_matrices=None, ages=None):
        """AI model: All detected cases incur monitoring and treatment costs"""
        years = len(cohort)
        costs = np.zeros(years)
        qalys = np.zeros(years)
        costs_discounted = np.zeros(years)
        qalys_discounted = np.zeros(years)
        screening_costs = np.zeros(years)

        cost_discount_rate = sample['cost_discount']
        health_discount_rate = sample['health_discount']

        # AI Model: ALL living states incur costs when detected (comprehensive care)
        state_costs = [
            sample['monitoring_mild'] + sample['treatment_mild'] + sample['other_mild'] + sample['productivity_mild'],
            sample['monitoring_moderate'] + sample['treatment_moderate'] + sample['other_moderate'] + sample['productivity_moderate'],
            sample['monitoring_severe'] + sample['treatment_severe'] + sample['other_severe'] + sample['productivity_severe'],
            sample['monitoring_vi'] + sample['treatment_vi'] + sample['other_vi'] + sample['productivity_vi']
        ]

        state_utilities = [
            sample['utility_mild'],
            sample['utility_moderate'],
            sample['utility_severe'],
            sample['utility_vi']
        ]

        if include_screening:
            annual_screening_cost = sample['ai_screening']
            detection_multiplier = sample['detection_proportion']
        else:
            annual_screening_cost = 0
            detection_multiplier = 1.0

        for year in range(years):
            # Only apply costs to living states (first 4 states)
            costs[year] = np.sum(cohort[year][:4] * state_costs) * detection_multiplier
            # Only living patients accumulate QALYs
            qalys[year] = np.sum(cohort[year][:4] * state_utilities)

            if include_screening and year == 0:
                screening_costs[year] = annual_screening_cost
                costs[year] += screening_costs[year]

            cost_discount_factor = 1 / (1 + cost_discount_rate) ** year
            health_discount_factor = 1 / (1 + health_discount_rate) ** year
            costs_discounted[year] = costs[year] * cost_discount_factor
            qalys_discounted[year] = qalys[year] * health_discount_factor

        return costs, qalys, costs_discounted, qalys_discounted, state_costs, state_utilities, screening_costs

    def run_deterministic(self, initial_dist=None, years=10, include_screening=True, 
                         population_type='general', starting_age=None):
        """Run AI model with mean parameter values"""
        if starting_age is not None:
            self.starting_age = starting_age
            
        if initial_dist is None:
            initial_dist = [1, 0, 0, 0, 0]  # All start in Mild, none dead

        sample = {}
        for category in [self.params.costs, self.params.utilities, self.params.transitions,
                        self.params.screening_accuracy, self.params.screening_accuracy_mild,
                        self.params.screening_accuracy_moderate, self.params.screening_accuracy_severe,
                        self.params.screening_params, self.params.discount_rates]:
            for name, param in category.items():
                sample[name] = param.mean

        # Add mortality multiplier parameters if they exist
        if hasattr(self.params, 'mortality_multipliers'):
            for name, param in self.params.mortality_multipliers.items():
                sample[name] = param.mean

        cohort, transition_matrices, ages = self.simulate_cohort(initial_dist, years, sample)
        costs, qalys, costs_disc, qalys_disc, state_costs, state_utilities, screening_costs = self.calculate_outcomes(
            cohort, sample, include_screening, population_type, transition_matrices, ages)

        traces = self.create_detailed_traces(cohort, costs, qalys, costs_disc, qalys_disc,
                                           state_costs, state_utilities, sample, screening_costs,
                                           transition_matrices, ages)

        return {
            'cohort': cohort,
            'costs': costs,
            'qalys': qalys,
            'costs_discounted': costs_disc,
            'qalys_discounted': qalys_disc,
            'total_cost': np.sum(costs),
            'total_qalys': np.sum(qalys),
            'total_cost_discounted': np.sum(costs_disc),
            'total_qalys_discounted': np.sum(qalys_disc),
            'traces': traces,
            'state_costs': state_costs,
            'state_utilities': state_utilities,
            'screening_costs': screening_costs,
            'sample_params': sample,
            'scenario_name': self.scenario_name,
            'model_type': self.model_type,
            'starting_age': self.starting_age,
            'transition_matrices': transition_matrices,
            'ages': ages
        }

    def run_probabilistic(self, n_iterations=1000, initial_dist=None, years=10,
                         include_screening=True, population_type='general', random_seed=42, 
                         return_traces=False, starting_age=None):
        """Run AI model probabilistic sensitivity analysis"""
        if starting_age is not None:
            self.starting_age = starting_age
            
        if initial_dist is None:
            initial_dist = [1, 0, 0, 0, 0]  # All start in Mild, none dead

        np.random.seed(random_seed)

        results = {
            'total_costs': [],
            'total_qalys': [],
            'total_costs_discounted': [],
            'total_qalys_discounted': [],
            'iterations': [],
            'parameters': [],
            'scenario_name': self.scenario_name,
            'model_type': self.model_type,
            'starting_age': self.starting_age
        }

        trace_tensor = None
        trace_variable_names = None

        if return_traces:
            # Define trace variables we want to store (now including Dead state, Age, and mortality rates)
            trace_vars = [
                'Year', 'Age', 'Prop_Mild', 'Prop_Moderate', 'Prop_Severe', 'Prop_VI', 'Prop_Dead', 'Prop_Alive',
                'Total_Cost', 'Total_QALY', 'Total_Cost_Disc', 'Total_QALY_Disc',
                'Screening_Cost', 'Cost_Mild', 'Cost_Moderate', 'Cost_Severe', 'Cost_VI',
                'QALY_Mild', 'QALY_Moderate', 'QALY_Severe', 'QALY_VI',
                'Cost_Discount_Factor', 'Health_Discount_Factor',
                'Mortality_Rate_Mild', 'Mortality_Rate_Moderate', 'Mortality_Rate_Severe', 'Mortality_Rate_VI',
                'Base_Mortality_qx'
            ]

            # Initialize 3D tensor: [iterations, years, variables]
            trace_tensor = np.zeros((n_iterations, years + 1, len(trace_vars)))
            trace_variable_names = trace_vars
            results['trace_variable_names'] = trace_variable_names

        print(f"Running {n_iterations} PSA iterations for AI Enhanced Model (Starting Age: {self.starting_age})...")

        for i in range(n_iterations):
            if (i + 1) % 100 == 0:
                print(f"  AI Model Iteration {i + 1}/{n_iterations}")

            sample = self.params.sample_all()
            cohort, transition_matrices, ages = self.simulate_cohort(initial_dist, years, sample)
            costs, qalys, costs_disc, qalys_disc, state_costs, state_utilities, screening_costs = self.calculate_outcomes(
                cohort, sample, include_screening, population_type, transition_matrices, ages)

            results['total_costs'].append(np.sum(costs))
            results['total_qalys'].append(np.sum(qalys))
            results['total_costs_discounted'].append(np.sum(costs_disc))
            results['total_qalys_discounted'].append(np.sum(qalys_disc))
            results['iterations'].append(i)
            results['parameters'].append(sample)

            # Store detailed traces if requested
            if return_traces:
                traces_df = self.create_detailed_traces(cohort, costs, qalys, costs_disc, qalys_disc,
                                                       state_costs, state_utilities, sample, screening_costs,
                                                       transition_matrices, ages)

                # Extract values for each trace variable and store in tensor
                for year_idx in range(years + 1):
                    year_data = traces_df.iloc[year_idx]
                    for var_idx, var_name in enumerate(trace_variable_names):
                        trace_tensor[i, year_idx, var_idx] = year_data[var_name]

        for key in ['total_costs', 'total_qalys', 'total_costs_discounted', 'total_qalys_discounted']:
            results[key] = np.array(results[key])

        results['trace_tensor'] = trace_tensor
        results['trace_variable_names'] = trace_variable_names

        return results


class NonAIGlaucomaModel(BaseGlaucomaModel):
    """Traditional/Non-AI Glaucoma Model with conventional screening and late detection"""

    def __init__(self, params=None, starting_age=60, mortality_table=None):
        if params is None:
            params = GlaucomaParameters.create_non_ai_pure_scenario()
        super().__init__(params, starting_age, mortality_table)
        self.model_type = "Traditional_NonAI"

    def calculate_outcomes(self, cohort, sample, include_screening=False, population_type='general',
                          transition_matrices=None, ages=None):
        """Non-AI model: Only VI patients incur costs (late detection model)"""
        years = len(cohort)
        costs = np.zeros(years)
        qalys = np.zeros(years)
        costs_discounted = np.zeros(years)
        qalys_discounted = np.zeros(years)
        screening_costs = np.zeros(years)

        cost_discount_rate = sample['cost_discount']
        health_discount_rate = sample['health_discount']

        # Non-AI Model: ONLY VI patients incur costs (early stages undetected)
        state_costs = [
            sample['monitoring_mild'] + sample['treatment_mild'] + sample['other_mild'] + sample['productivity_mild'],
            sample['monitoring_moderate'] + sample['treatment_moderate'] + sample['other_moderate'] + sample['productivity_moderate'],
            sample['monitoring_severe'] + sample['treatment_severe'] + sample['other_severe'] + sample['productivity_severe'],
            sample['monitoring_vi'] + sample['treatment_vi'] + sample['other_vi'] + sample['productivity_vi']
        ]

        state_utilities = [
            sample['utility_mild'],
            sample['utility_moderate'],
            sample['utility_severe'],
            sample['utility_vi']
        ]

        if include_screening:
            annual_screening_cost = sample['human_screening']
            detection_multiplier = 1.0
        else:
            annual_screening_cost = 0
            detection_multiplier = 1.0

        for year in range(years):
            # Only apply costs to living states (first 4 states)
            costs[year] = np.sum(cohort[year][:4] * state_costs)
            # Only living patients accumulate QALYs
            qalys[year] = np.sum(cohort[year][:4] * state_utilities)

            if include_screening and year == 0:
                screening_costs[year] = annual_screening_cost
                costs[year] += screening_costs[year]

            cost_discount_factor = 1 / (1 + cost_discount_rate) ** year
            health_discount_factor = 1 / (1 + health_discount_rate) ** year
            costs_discounted[year] = costs[year] * cost_discount_factor
            qalys_discounted[year] = qalys[year] * health_discount_factor

        return costs, qalys, costs_discounted, qalys_discounted, state_costs, state_utilities, screening_costs

    def run_deterministic(self, initial_dist=None, years=10, include_screening=False, 
                         population_type='general', starting_age=None):
        """Run Non-AI model with mean parameter values"""
        if starting_age is not None:
            self.starting_age = starting_age
            
        if initial_dist is None:
            initial_dist = [1, 0, 0, 0, 0]  # All start in Mild, none dead

        sample = {}
        for category in [self.params.costs, self.params.utilities, self.params.transitions,
                        self.params.screening_accuracy, self.params.screening_accuracy_mild,
                        self.params.screening_accuracy_moderate, self.params.screening_accuracy_severe,
                        self.params.screening_params, self.params.discount_rates]:
            for name, param in category.items():
                sample[name] = param.mean

        # Add mortality multiplier parameters if they exist
        if hasattr(self.params, 'mortality_multipliers'):
            for name, param in self.params.mortality_multipliers.items():
                sample[name] = param.mean

        cohort, transition_matrices, ages = self.simulate_cohort(initial_dist, years, sample)
        costs, qalys, costs_disc, qalys_disc, state_costs, state_utilities, screening_costs = self.calculate_outcomes(
            cohort, sample, include_screening, population_type, transition_matrices, ages)

        traces = self.create_detailed_traces(cohort, costs, qalys, costs_disc, qalys_disc,
                                           state_costs, state_utilities, sample, screening_costs,
                                           transition_matrices, ages)

        return {
            'cohort': cohort,
            'costs': costs,
            'qalys': qalys,
            'costs_discounted': costs_disc,
            'qalys_discounted': qalys_disc,
            'total_cost': np.sum(costs),
            'total_qalys': np.sum(qalys),
            'total_cost_discounted': np.sum(costs_disc),
            'total_qalys_discounted': np.sum(qalys_disc),
            'traces': traces,
            'state_costs': state_costs,
            'state_utilities': state_utilities,
            'screening_costs': screening_costs,
            'sample_params': sample,
            'scenario_name': self.scenario_name,
            'model_type': self.model_type,
            'starting_age': self.starting_age,
            'transition_matrices': transition_matrices,
            'ages': ages
        }

    def run_probabilistic(self, n_iterations=1000, initial_dist=None, years=10,
                         include_screening=False, population_type='general', random_seed=42, 
                         return_traces=False, starting_age=None):
        """Run Non-AI model probabilistic sensitivity analysis"""
        if starting_age is not None:
            self.starting_age = starting_age
            
        if initial_dist is None:
            initial_dist = [1, 0, 0, 0, 0]  # All start in Mild, none dead

        np.random.seed(random_seed)

        results = {
            'total_costs': [],
            'total_qalys': [],
            'total_costs_discounted': [],
            'total_qalys_discounted': [],
            'iterations': [],
            'parameters': [],
            'scenario_name': self.scenario_name,
            'model_type': self.model_type,
            'starting_age': self.starting_age
        }

        trace_tensor = None
        trace_variable_names = None

        if return_traces:
            # Define trace variables we want to store (now including Dead state, Age, and mortality rates)
            trace_vars = [
                'Year', 'Age', 'Prop_Mild', 'Prop_Moderate', 'Prop_Severe', 'Prop_VI', 'Prop_Dead', 'Prop_Alive',
                'Total_Cost', 'Total_QALY', 'Total_Cost_Disc', 'Total_QALY_Disc',
                'Screening_Cost', 'Cost_Mild', 'Cost_Moderate', 'Cost_Severe', 'Cost_VI',
                'QALY_Mild', 'QALY_Moderate', 'QALY_Severe', 'QALY_VI',
                'Cost_Discount_Factor', 'Health_Discount_Factor',
                'Mortality_Rate_Mild', 'Mortality_Rate_Moderate', 'Mortality_Rate_Severe', 'Mortality_Rate_VI',
                'Base_Mortality_qx'
            ]

            # Initialize 3D tensor: [iterations, years, variables]
            trace_tensor = np.zeros((n_iterations, years + 1, len(trace_vars)))
            trace_variable_names = trace_vars
            results['trace_variable_names'] = trace_variable_names

        print(f"Running {n_iterations} PSA iterations for Traditional Non-AI Model (Starting Age: {self.starting_age})...")

        for i in range(n_iterations):
            if (i + 1) % 100 == 0:
                print(f"  Non-AI Model Iteration {i + 1}/{n_iterations}")

            sample = self.params.sample_all()
            cohort, transition_matrices, ages = self.simulate_cohort(initial_dist, years, sample)
            costs, qalys, costs_disc, qalys_disc, state_costs, state_utilities, screening_costs = self.calculate_outcomes(
                cohort, sample, include_screening, population_type, transition_matrices, ages)

            results['total_costs'].append(np.sum(costs))
            results['total_qalys'].append(np.sum(qalys))
            results['total_costs_discounted'].append(np.sum(costs_disc))
            results['total_qalys_discounted'].append(np.sum(qalys_disc))
            results['iterations'].append(i)
            results['parameters'].append(sample)

            # Store detailed traces if requested
            if return_traces:
                traces_df = self.create_detailed_traces(cohort, costs, qalys, costs_disc, qalys_disc,
                                                       state_costs, state_utilities, sample, screening_costs,
                                                       transition_matrices, ages)

                # Extract values for each trace variable and store in tensor
                for year_idx in range(years + 1):
                    year_data = traces_df.iloc[year_idx]
                    for var_idx, var_name in enumerate(trace_variable_names):
                        trace_tensor[i, year_idx, var_idx] = year_data[var_name]

        for key in ['total_costs', 'total_qalys', 'total_costs_discounted', 'total_qalys_discounted']:
            results[key] = np.array(results[key])

        results['trace_tensor'] = trace_tensor
        results['trace_variable_names'] = trace_variable_names

        return results


# Comparison and utility functions (same as before)
def compare_ai_vs_nonai_models(results_ai, results_non_ai, discounted=True):
    """Compare AI vs Non-AI model results and calculate incremental metrics"""
    if discounted:
        costs_ai = results_ai['total_costs_discounted']
        qalys_ai = results_ai['total_qalys_discounted']
        costs_non_ai = results_non_ai['total_costs_discounted']
        qalys_non_ai = results_non_ai['total_qalys_discounted']
    else:
        costs_ai = results_ai['total_costs']
        qalys_ai = results_ai['total_qalys']
        costs_non_ai = results_non_ai['total_costs']
        qalys_non_ai = results_non_ai['total_qalys']

    incremental_costs = costs_ai - costs_non_ai
    incremental_qalys = qalys_ai - qalys_non_ai

    icer_values = np.where(incremental_qalys != 0,
                          incremental_costs / incremental_qalys,
                          np.inf)

    comparison = {
        'incremental_costs_mean': np.mean(incremental_costs),
        'incremental_costs_std': np.std(incremental_costs),
        'incremental_qalys_mean': np.mean(incremental_qalys),
        'incremental_qalys_std': np.std(incremental_qalys),
        'icer_mean': np.mean(icer_values[np.isfinite(icer_values)]),
        'icer_std': np.std(icer_values[np.isfinite(icer_values)]),
        'costs_ai_mean': np.mean(costs_ai),
        'qalys_ai_mean': np.mean(qalys_ai),
        'costs_non_ai_mean': np.mean(costs_non_ai),
        'qalys_non_ai_mean': np.mean(qalys_non_ai),
        'ai_model_type': results_ai.get('model_type', 'AI'),
        'non_ai_model_type': results_non_ai.get('model_type', 'Non-AI'),
        'discounted': discounted
    }

    return comparison


def run_full_ai_vs_nonai_analysis(years=10, n_iterations=1000, return_traces=False, 
                                   starting_age=60, mortality_table=None):
    """Run complete analysis comparing separate AI and Non-AI models"""

    print("=== Running Full AI vs Non-AI Model Comparison ===")

    ai_model = AIGlaucomaModel(starting_age=starting_age, mortality_table=mortality_table)
    nonai_model = NonAIGlaucomaModel(starting_age=starting_age, mortality_table=mortality_table)

    print(f"AI Model: {ai_model.model_type} (Starting Age: {starting_age})")
    print(f"Non-AI Model: {nonai_model.model_type} (Starting Age: {starting_age})")

    print("\n1. Running deterministic analyses...")
    det_ai = ai_model.run_deterministic(years=years)
    det_nonai = nonai_model.run_deterministic(years=years)

    print("\n2. Running probabilistic analyses...")
    psa_ai = ai_model.run_probabilistic(n_iterations=n_iterations, years=years, return_traces=return_traces)
    psa_nonai = nonai_model.run_probabilistic(n_iterations=n_iterations, years=years, return_traces=return_traces)

    print("\n3. Comparing results...")
    comparison = compare_ai_vs_nonai_models(psa_ai, psa_nonai, discounted=True)

    return {
        'ai_model': ai_model,
        'nonai_model': nonai_model,
        'deterministic_ai': det_ai,
        'deterministic_nonai': det_nonai,
        'probabilistic_ai': psa_ai,
        'probabilistic_nonai': psa_nonai,
        'comparison': comparison
    }


def quick_model_comparison(starting_age=60, mortality_table=None):
    """Quick 2-line comparison of AI vs Non-AI models"""
    ai_model = AIGlaucomaModel(starting_age=starting_age, mortality_table=mortality_table)
    nonai_model = NonAIGlaucomaModel(starting_age=starting_age, mortality_table=mortality_table)

    ai_results = ai_model.run_deterministic()
    nonai_results = nonai_model.run_deterministic()

    print(f"Starting Age: {starting_age}")
    print(f"AI Model Total Cost (Discounted): ${ai_results['total_cost_discounted']:,.0f}")
    print(f"AI Model Total QALYs (Discounted): {ai_results['total_qalys_discounted']:.2f}")
    print(f"Non-AI Model Total Cost (Discounted): ${nonai_results['total_cost_discounted']:,.0f}")
    print(f"Non-AI Model Total QALYs (Discounted): {nonai_results['total_qalys_discounted']:.2f}")

    return ai_results, nonai_results


# Utility functions for working with trace tensors (same as before)
def get_trace_summary_stats(trace_tensor, trace_variable_names, variable_name, year=None):
    """Get summary statistics for a specific variable across all simulations"""
    var_idx = trace_variable_names.index(variable_name)

    if year is not None:
        data = trace_tensor[:, year, var_idx]
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'median': np.median(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'min': np.min(data),
            'max': np.max(data)
        }
    else:
        data = trace_tensor[:, :, var_idx]
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'median': np.median(data, axis=0),
            'q25': np.percentile(data, 25, axis=0),
            'q75': np.percentile(data, 75, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0)
        }


def get_trace_percentiles(trace_tensor, trace_variable_names, variable_name, percentiles=[5, 25, 50, 75, 95]):
    """Get specified percentiles for a variable across all years and simulations"""
    var_idx = trace_variable_names.index(variable_name)
    data = trace_tensor[:, :, var_idx]

    result = {}
    for p in percentiles:
        result[f'p{p}'] = np.percentile(data, p, axis=0)

    return result


def extract_trace_variable(trace_tensor, trace_variable_names, variable_name):
    """Extract a specific variable from the trace tensor"""
    var_idx = trace_variable_names.index(variable_name)
    return trace_tensor[:, :, var_idx]


def compare_trace_variables(trace_tensor_ai, trace_tensor_nonai, trace_variable_names, variable_name):
    """Compare a specific variable between AI and Non-AI models"""
    ai_data = extract_trace_variable(trace_tensor_ai, trace_variable_names, variable_name)
    nonai_data = extract_trace_variable(trace_tensor_nonai, trace_variable_names, variable_name)

    incremental = ai_data - nonai_data

    return {
        'ai_mean': np.mean(ai_data, axis=0),
        'nonai_mean': np.mean(nonai_data, axis=0),
        'incremental_mean': np.mean(incremental, axis=0),
        'incremental_std': np.std(incremental, axis=0),
        'incremental_median': np.median(incremental, axis=0),
        'incremental_q25': np.percentile(incremental, 25, axis=0),
        'incremental_q75': np.percentile(incremental, 75, axis=0)
    }


def create_trace_dataframe(trace_tensor, trace_variable_names, iteration=0):
    """Convert a single iteration's trace tensor to a DataFrame for easy viewing"""
    data = trace_tensor[iteration, :, :]
    return pd.DataFrame(data, columns=trace_variable_names)


print("Enhanced models with age-dependent mortality from life tables defined!")