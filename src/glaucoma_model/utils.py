# src/glaucoma_model/utils.py
import pickle
import copy
from pathlib import Path
import pandas as pd
import numpy as np


def save_base_case(ai_results, non_ai_results, model_params_dict, variable_names, original_prevalence, filepath='../../data/base_case.pkl'):
    """
    Save base case results and model parameters
    
    Parameters:
    - ai_results: AI model results dictionary
    - non_ai_results: Non-AI model results dictionary
    - model_params_dict: Dictionary of model parameters (extracted from model_ai.params)
    - variable_names: list of variable names
    - original_prevalence: the prevalence value used in the base case
    - filepath: path to save the pickle file
    """
    base_case_data = {
        'ai_results': ai_results,
        'non_ai_results': non_ai_results,
        'model_params': model_params_dict,
        'variable_names': variable_names,
        'original_prevalence': original_prevalence
    }
    
    current_dir = Path(__file__).parent
    filepath = (current_dir / filepath).resolve()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(base_case_data, f)
    
    print(f"Base case saved to {filepath}")


def extract_model_params(model):
    """
    Extract serializable parameters from model object
    
    Parameters:
    - model: the model object
    
    Returns:
    - Dictionary of parameters
    """
    return {
        'costs': {k: {'mean': v.mean, 'std': getattr(v, 'std', None)} 
                  for k, v in model.params.costs.items()},
        'screening_params': {k: {'mean': v.mean, 'std': getattr(v, 'std', None)} 
                            for k, v in model.params.screening_params.items()},
        'screening_accuracy': {k: {'mean': v.mean, 'std': getattr(v, 'std', None)} 
                              for k, v in model.params.screening_accuracy.items()},
        # Add other parameter groups as needed
    }


def load_base_case(filepath='../../data/base_case.pkl'):
    """
    Load base case results and model parameters
    
    Parameters:
    - filepath: path to the pickle file
    
    Returns:
    - Dictionary containing ai_results, non_ai_results, model_params, variable_names, original_prevalence
    """
    current_dir = Path(__file__).parent
    filepath = (current_dir / filepath).resolve()
    
    with open(filepath, 'rb') as f:
        base_case_data = pickle.load(f)
    
    print(f"Base case loaded from {filepath}")
    return base_case_data

def create_prevalence_scenario(base_results, model_params, prevalence_value, variable_names, original_prevalence):
    """
    Create a scenario with a specific prevalence value
    
    Correctly accounts for prevalence-dependent screening costs:
    - AI screening cost: (1/prevalence) × AI_cost_per_person
    - Human follow-up cost: ((1-prevalence)/prevalence) × (1-specificity) × human_cost_per_person
    
    Parameters:
    - base_results: original results dictionary
    - model_params: dictionary of model parameters (from extract_model_params)
    - prevalence_value: the new prevalence to test
    - variable_names: list of variable names
    - original_prevalence: the prevalence used in the base case
    
    Returns:
    - Modified copy of results with adjusted costs
    """
    scenario_results = copy.deepcopy(base_results)
    
    idx1 = variable_names.index('Total_Cost')
    idx2 = variable_names.index('Total_Cost_Disc')
    
    # Get parameters
    ai_screening_cost = model_params['costs']['ai_screening'].mean
    print("ai screening cost ",ai_screening_cost )
    human_screening_cost = model_params['costs']['human_screening'].mean
    print("human costs", human_screening_cost)
    specificity = model_params['screening_accuracy']['specificity'].mean
    
    # ORIGINAL scenario costs
    # AI screening: number of people to screen × cost per person
    original_ai_cost = (1/original_prevalence) * ai_screening_cost
    
    # Human follow-up: number of false positives × cost per person
    # False positives per case = ((1-prevalence)/prevalence) × (1-specificity)
    original_false_positives = ((1 - original_prevalence) / original_prevalence) * (1 - specificity)
    original_human_cost = original_false_positives * human_screening_cost
    
    # NEW scenario costs
    new_ai_cost = (1/prevalence_value) * ai_screening_cost
    new_false_positives = ((1 - prevalence_value) / prevalence_value) * (1 - specificity)
    new_human_cost = new_false_positives * human_screening_cost
    
    # Calculate total adjustment (difference between new and original)
    total_cost_adjustment = (new_ai_cost + new_human_cost) - (original_ai_cost + original_human_cost)
    print("total_cost_adjustment ",total_cost_adjustment)
    # Apply to first time period only (year 0)
    scenario_results['trace_tensor'][:,0,idx1] += total_cost_adjustment
    scenario_results['trace_tensor'][:,0,idx2] += total_cost_adjustment
    
    return scenario_results


def run_prevalence_sensitivity_analysis(
    ai_base_results, 
    non_ai_base_results,
    model_params,
    variable_names,
    original_prevalence,
    prevalence_values,
    analysis_function,
    **analysis_kwargs
):
    """
    Run cost-effectiveness analysis across multiple prevalence scenarios
    
    Parameters:
    - ai_base_results: base AI results dictionary
    - non_ai_base_results: base non-AI results dictionary
    - model_params: dictionary of model parameters
    - variable_names: list of variable names
    - original_prevalence: the prevalence value used in the base case
    - prevalence_values: list of prevalence values to test
    - analysis_function: the comprehensive_cost_effectiveness_analysis function
    - **analysis_kwargs: additional arguments for analysis_function (NOT including original_prevalence)
    
    Returns:
    - Dictionary with prevalence values as keys and comprehensive analysis results as values
    """
    sensitivity_results = {}
    
    for prev in prevalence_values:
        print(f"\n{'='*60}")
        print(f"Analyzing prevalence = {prev*100}%")
        print(f"{'='*60}")
        
        # Create scenario-specific results
        ai_scenario = create_prevalence_scenario(ai_base_results, model_params, prev, variable_names, original_prevalence)
        non_ai_scenario = non_ai_base_results
        
        # Run comprehensive analysis (analysis_kwargs should NOT contain original_prevalence)
        results = analysis_function(ai_scenario, non_ai_scenario, **analysis_kwargs)
        
        sensitivity_results[prev] = results
    
    return sensitivity_results


def summarize_prevalence_sensitivity(sensitivity_results, time_horizon='10_years'):
    """
    Create a summary table comparing results across prevalence values
    
    Parameters:
    - sensitivity_results: output from run_prevalence_sensitivity_analysis
    - time_horizon: which time horizon to summarize (e.g., '10_years', '20_years')
    
    Returns:
    - DataFrame with summary statistics across prevalence values
    """
    summary_data = []
    
    for prev, results in sensitivity_results.items():
        if time_horizon in results:
            summary = results[time_horizon]['summary']
            summary_data.append({
                'Prevalence': f"{prev*100}%",
                'Prevalence_Value': prev,
                'Incremental_Cost': summary['incremental_cost_mean'],
                'Incremental_Cost_95CI': f"[€{summary['incremental_cost_ci'][0]:,.0f}, €{summary['incremental_cost_ci'][1]:,.0f}]",
                'Incremental_QALY': summary['incremental_qaly_mean'],
                'Incremental_QALY_95CI': f"[{summary['incremental_qaly_ci'][0]:.3f}, {summary['incremental_qaly_ci'][1]:.3f}]",
                'ICER': summary['incremental_cost_mean'] / summary['incremental_qaly_mean'] if summary['incremental_qaly_mean'] != 0 else np.nan,
                'ICER_Mean': summary['icer_mean'],
                'ICER_Median': summary['icer_median'],
                'ICER_95CI': f"[€{summary['icer_ci'][0]:,.0f}, €{summary['icer_ci'][1]:,.0f}]" if np.isfinite(summary['icer_ci'][0]) else "N/A",
                'Prob_CE_20k': summary['prob_cost_effective_20k'],
                'Prob_CE_50k': summary['prob_cost_effective_50k'],
                'Prob_CE_100k': summary['prob_cost_effective_100k'],
            })
    
    return pd.DataFrame(summary_data)


def save_scenario_results(scenario_results, scenario_name, output_dir='../../data/scenarios'):
    """
    Save scenario analysis results
    
    Parameters:
    - scenario_results: results from run_prevalence_sensitivity_analysis
    - scenario_name: name for this scenario
    - output_dir: directory to save results (relative to project root)
    """
    current_dir = Path(__file__).parent
    output_path = (current_dir / output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / f"{scenario_name}.pkl"
    
    with open(filepath, 'wb') as f:
        pickle.dump(scenario_results, f)
    
    print(f"Scenario results saved to {filepath}")


def load_scenario_results(scenario_name, output_dir='../../data/scenarios'):
    """
    Load scenario analysis results
    
    Parameters:
    - scenario_name: name of the scenario
    - output_dir: directory containing results
    
    Returns:
    - Scenario results dictionary
    """
    current_dir = Path(__file__).parent
    filepath = (current_dir / output_dir).resolve() / f"{scenario_name}.pkl"
    
    with open(filepath, 'rb') as f:
        scenario_results = pickle.load(f)
    
    print(f"Scenario results loaded from {filepath}")
    return scenario_results

def create_mixed_scenarios(ai_results, non_ai_results, variable_names, mix_ratio_1=0.5, mix_ratio_2=0.5):
    """
    Create two mixed scenarios from AI and non-AI base scenarios
    """
    # Create deep copies
    mixed_scenario_1 = copy.deepcopy(ai_results)
    mixed_scenario_2 = copy.deepcopy(ai_results)
    
    # Extract trace arrays
    ai_traces = ai_results['trace_tensor']
    non_ai_traces = non_ai_results['trace_tensor']
    
    # Define variables to adjust
    cost_variables = ['Total_Cost', 'Total_Cost_Disc']
    qaly_variables = ['Total_QALY', 'Total_QALY_Disc',
                      'QALY_Mild', 'QALY_Moderate', 'QALY_Severe', 'QALY_VI']
    
    variables_to_adjust = cost_variables + qaly_variables
    indices = [variable_names.index(var) for var in variables_to_adjust]
    
    # Create new trace tensors
    mixed_traces_1 = np.copy(ai_traces)
    mixed_traces_2 = np.copy(ai_traces)
    
    # Mix the variables
    for idx in indices:
        mixed_traces_1[:, :, idx] = (
            mix_ratio_1 * ai_traces[:, :, idx] + 
            (1 - mix_ratio_1) * non_ai_traces[:, :, idx]
        )
        mixed_traces_2[:, :, idx] = (
            mix_ratio_2 * ai_traces[:, :, idx] + 
            (1 - mix_ratio_2) * non_ai_traces[:, :, idx]
        )
    
    # Update
    mixed_scenario_1['trace_tensor'] = mixed_traces_1
    mixed_scenario_2['trace_tensor'] = mixed_traces_2
    
    # DIAGNOSTIC PRINTS
    cost_idx = variable_names.index('Total_Cost_Disc')
    qaly_idx = variable_names.index('Total_QALY_Disc')
    
    print(f"\n=== DIAGNOSTIC: Scenario Creation ===")
    print(f"Mix ratios: {mix_ratio_1} vs {mix_ratio_2}")
    print(f"\nFirst simulation, all years, Total_Cost_Disc:")
    print(f"  AI sum: {np.sum(ai_traces[0, :, cost_idx]):.2f}")
    print(f"  Non-AI sum: {np.sum(non_ai_traces[0, :, cost_idx]):.2f}")
    print(f"  Mixed 1 sum: {np.sum(mixed_traces_1[0, :, cost_idx]):.2f}")
    print(f"  Mixed 2 sum: {np.sum(mixed_traces_2[0, :, cost_idx]):.2f}")
    
    print(f"\nFirst simulation, all years, Total_QALY_Disc:")
    print(f"  AI sum: {np.sum(ai_traces[0, :, qaly_idx]):.3f}")
    print(f"  Non-AI sum: {np.sum(non_ai_traces[0, :, qaly_idx]):.3f}")
    print(f"  Mixed 1 sum: {np.sum(mixed_traces_1[0, :, qaly_idx]):.3f}")
    print(f"  Mixed 2 sum: {np.sum(mixed_traces_2[0, :, qaly_idx]):.3f}")
    
    # Check if scenarios are actually different
    cost_diff = np.sum(mixed_traces_1[:, :, cost_idx]) - np.sum(mixed_traces_2[:, :, cost_idx])
    qaly_diff = np.sum(mixed_traces_1[:, :, qaly_idx]) - np.sum(mixed_traces_2[:, :, qaly_idx])
    print(f"\nDifference between mixed scenarios (all sims):")
    print(f"  Total cost difference: {cost_diff:.2f}")
    print(f"  Total QALY difference: {qaly_diff:.3f}")
    
    return mixed_scenario_1, mixed_scenario_2