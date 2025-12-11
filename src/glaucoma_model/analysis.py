import numpy as np
import pandas as pd
from scipy import stats
def calculate_icer_with_time_horizon(ai_results, nonai_results, time_horizon=None, discounted=True, icer_limits=(-500000, 500000)):
    """
    Calculate ICER with flexible time horizon
    """
    print(">>> USING NEW VERSION WITH LOOP <<<")
    
    # Extract trace tensors
    ai_traces = ai_results['trace_tensor'].copy()
    nonai_traces = nonai_results['trace_tensor'].copy()
    variable_names = ai_results['trace_variable_names']

    # Get cost and QALY variable indices
    if discounted:
        cost_var = 'Total_Cost_Disc'
        qaly_var = 'Total_QALY_Disc'
    else:
        cost_var = 'Total_Cost'
        qaly_var = 'Total_QALY'

    cost_idx = variable_names.index(cost_var)
    qaly_idx = variable_names.index(qaly_var)

    # Determine years to include
    max_years = ai_traces.shape[1]
    if time_horizon is None:
        years_to_include = max_years
    else:
        years_to_include = min(time_horizon + 1, max_years)

    # Calculate total costs and QALYs
    n_iterations = ai_traces.shape[0]

    ai_total_costs = np.sum(ai_traces[:, :years_to_include, cost_idx], axis=1)
    ai_total_qalys = np.sum(ai_traces[:, :years_to_include, qaly_idx], axis=1)

    nonai_total_costs = np.sum(nonai_traces[:, :years_to_include, cost_idx], axis=1)
    nonai_total_qalys = np.sum(nonai_traces[:, :years_to_include, qaly_idx], axis=1)

    # Calculate incremental values
    incremental_costs = np.array(ai_total_costs - nonai_total_costs)
    incremental_qalys = np.array(ai_total_qalys - nonai_total_qalys)

    # Calculate ICER
    icers = np.zeros(len(incremental_costs))
    for i in range(len(incremental_costs)):
        if incremental_qalys[i] != 0:
            icers[i] = incremental_costs[i] / incremental_qalys[i]
        else:
            icers[i] = np.inf
    
    # Create combined filter
    finite_mask = np.isfinite(icers)
    limits_mask = (icers >= icer_limits[0]) & (icers <= icer_limits[1])
    combined_mask = finite_mask & limits_mask
    
    # Apply mask
    incremental_costs_filtered = incremental_costs[combined_mask]
    incremental_qalys_filtered = incremental_qalys[combined_mask]
    icers_filtered = icers[combined_mask]
    
    print(f"Number of ICERs after filtering: {len(icers_filtered)} out of {n_iterations}")
    print(f"Mean ICER: {np.mean(icers_filtered):.2f}")

    # THE RETURN MUST BE THE LAST THING IN THE FUNCTION
    return {
        'time_horizon': time_horizon if time_horizon is not None else max_years - 1,
        'years_included': years_to_include - 1,
        'n_simulations': n_iterations,
        'discounted': discounted,
        'incremental_costs': incremental_costs_filtered,
        'incremental_costs_mean': np.mean(incremental_costs_filtered),
        'incremental_costs_std': np.std(incremental_costs_filtered),
        'incremental_costs_median': np.median(incremental_costs_filtered),
        'incremental_qalys': incremental_qalys_filtered,
        'incremental_qalys_mean': np.mean(incremental_qalys_filtered),
        'incremental_qalys_std': np.std(incremental_qalys_filtered),
        'incremental_qalys_median': np.median(incremental_qalys_filtered),
        'icers': icers,
        'finite_icers': icers_filtered,
        'icer_mean': np.mean(icers_filtered) if len(icers_filtered) > 0 else np.inf,
        'icer_median': np.median(icers_filtered) if len(icers_filtered) > 0 else np.inf,
        'ai_mean_cost': np.mean(ai_total_costs),
        'ai_mean_qalys': np.mean(ai_total_qalys),
        'nonai_mean_cost': np.mean(nonai_total_costs),
        'nonai_mean_qalys': np.mean(nonai_total_qalys),
    }
    
    # ... rest of the function ...
def calculate_icer_confidence_intervals(icer_results, confidence_level=0.95):
    """
    Calculate confidence intervals for ICER using bootstrap or percentile method

    Parameters:
    - icer_results: Results from calculate_icer_with_time_horizon
    - confidence_level: Confidence level (default 0.95 for 95% CI)

    Returns:
    - Dictionary with confidence intervals
    """

    incremental_costs = icer_results['incremental_costs']
    incremental_qalys = icer_results['incremental_qalys']
    finite_icers = icer_results['finite_icers']


    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100

    # ICER confidence interval using percentile method
    if len(finite_icers) > 0:
        icer_ci_lower = np.percentile(finite_icers, lower_percentile)
        icer_ci_upper = np.percentile(finite_icers, upper_percentile)
    else:
        icer_ci_lower = np.inf
        icer_ci_upper = np.inf

    # Incremental costs CI
    cost_ci_lower = np.percentile(incremental_costs, lower_percentile)
    cost_ci_upper = np.percentile(incremental_costs, upper_percentile)

    # Incremental QALYs CI
    qaly_ci_lower = np.percentile(incremental_qalys, lower_percentile)
    qaly_ci_upper = np.percentile(incremental_qalys, upper_percentile)

    return {
        'confidence_level': confidence_level,
        'icer_ci_lower': icer_ci_lower,
        'icer_ci_upper': icer_ci_upper,
        'incremental_costs_ci_lower': cost_ci_lower,
        'incremental_costs_ci_upper': cost_ci_upper,
        'incremental_qalys_ci_lower': qaly_ci_lower,
        'incremental_qalys_ci_upper': qaly_ci_upper,
    }


def probability_cost_effective_multiple_thresholds(icer_results, thresholds=None):
    """
    Calculate probability of cost-effectiveness at multiple ICER thresholds

    Parameters:
    - icer_results: Results from calculate_icer_with_time_horizon
    - thresholds: List of ICER thresholds (€/QALY)

    Returns:
    - DataFrame with probabilities for each threshold
    """

    if thresholds is None:
        thresholds = [0, 25000, 50000, 75000, 100000, 150000, 200000]

    incremental_costs = icer_results['incremental_costs']
    incremental_qalys = icer_results['incremental_qalys']

    results = []

    for threshold in thresholds:
        # Cost-effective if:
        # 1. Dominates: lower cost AND higher QALYs
        # 2. ICER below threshold: higher cost AND higher QALYs AND ICER <= threshold
        # 3. Saves money with acceptable QALY loss: lower cost (regardless of QALY change)

        dominates = (incremental_costs < 0) & (incremental_qalys > 0)
        saves_money = incremental_costs < 0

        cost_effective_positive = (incremental_costs > 0) & (incremental_qalys > 0) & \
                                 ((incremental_costs / incremental_qalys) <= threshold)

        # Different definitions of cost-effectiveness
        cost_effective_strict = dominates | cost_effective_positive  # Only if QALY gain
        cost_effective_relaxed = saves_money | cost_effective_positive  # Include cost savings

        prob_strict = np.sum(cost_effective_strict) / len(cost_effective_strict)
        prob_relaxed = np.sum(cost_effective_relaxed) / len(cost_effective_relaxed)

        results.append({
            'threshold': threshold,
            'prob_cost_effective_strict': prob_strict,
            'prob_cost_effective_relaxed': prob_relaxed,
            'n_dominates': np.sum(dominates),
            'n_saves_money': np.sum(saves_money),
            'n_cost_effective_positive': np.sum(cost_effective_positive),
            'percent_dominates': np.sum(dominates) / len(dominates) * 100,
            'percent_saves_money': np.sum(saves_money) / len(saves_money) * 100,
            'percent_cost_effective_strict': prob_strict * 100,
            'percent_cost_effective_relaxed': prob_relaxed * 100,
        })

    return pd.DataFrame(results)


def comprehensive_cost_effectiveness_analysis(ai_results, nonai_results,
                                            time_horizons=[5, 10],
                                            thresholds=None,
                                            confidence_level=0.95,
                                            discounted=True):
    """
    Comprehensive cost-effectiveness analysis with multiple time horizons

    Parameters:
    - ai_results, nonai_results: Model results with traces
    - time_horizons: List of time horizons to analyze
    - thresholds: ICER thresholds to test
    - confidence_level: For confidence intervals
    - discounted: Use discounted values

    Returns:
    - Dictionary with complete analysis results
    """

    if thresholds is None:
        thresholds = [0, 20000, 50000, 75000, 100000, 150000, 200000]

    results = {}

    for horizon in time_horizons:
        print(f"Analyzing {horizon}-year time horizon...")

        # Calculate ICER for this time horizon
        icer_results = calculate_icer_with_time_horizon(
            ai_results, nonai_results, time_horizon=horizon, discounted=discounted
        )

        # Calculate confidence intervals
        ci_results = calculate_icer_confidence_intervals(icer_results, confidence_level)

        # Calculate probabilities at different thresholds
        prob_results = probability_cost_effective_multiple_thresholds(icer_results, thresholds)

        # Store all results for this time horizon
        results[f'{horizon}_years'] = {
            'icer_analysis': icer_results,
            'confidence_intervals': ci_results,
            'probability_analysis': prob_results,
            'summary': {
                'time_horizon': horizon,
                'incremental_cost_mean': icer_results['incremental_costs_mean'],
                'incremental_cost_ci': (ci_results['incremental_costs_ci_lower'],
                                      ci_results['incremental_costs_ci_upper']),
                'incremental_qaly_mean': icer_results['incremental_qalys_mean'],
                'incremental_qaly_ci': (ci_results['incremental_qalys_ci_lower'],
                                      ci_results['incremental_qalys_ci_upper']),
                'icer_mean': icer_results['icer_mean'],
                'icer_median': icer_results['icer_median'],
                'icer_ci': (ci_results['icer_ci_lower'], ci_results['icer_ci_upper']),
                'prob_cost_effective_20k': prob_results[prob_results['threshold'] == 20000]['percent_cost_effective_relaxed'].iloc[0] if 20000 in prob_results['threshold'].values else None,
                'prob_cost_effective_50k': prob_results[prob_results['threshold'] == 50000]['percent_cost_effective_relaxed'].iloc[0] if 50000 in prob_results['threshold'].values else None,
                'prob_cost_effective_100k': prob_results[prob_results['threshold'] == 100000]['percent_cost_effective_relaxed'].iloc[0] if 100000 in prob_results['threshold'].values else None,
            }
        }

    return results


def create_summary_table(comprehensive_results):
    """
    Create a formatted summary table of cost-effectiveness results

    Parameters:
    - comprehensive_results: Results from comprehensive_cost_effectiveness_analysis

    Returns:
    - pandas DataFrame with formatted summary
    """

    summary_data = []

    for horizon_key, horizon_data in comprehensive_results.items():
        summary = horizon_data['summary']

        row = {
            'Time_Horizon': f"{summary['time_horizon']} years",
            'Incremental_Cost_Mean': f"€{summary['incremental_cost_mean']:,.0f}",
            'Incremental_Cost_95CI': f"[€{summary['incremental_cost_ci'][0]:,.0f}, €{summary['incremental_cost_ci'][1]:,.0f}]",
            'Incremental_QALY_Mean': f"{summary['incremental_qaly_mean']:.3f}",
            'Incremental_QALY_95CI': f"[{summary['incremental_qaly_ci'][0]:.3f}, {summary['incremental_qaly_ci'][1]:.3f}]",
            'ICER': f"€{summary['incremental_cost_mean'] / summary['incremental_qaly_mean']:,.0f}" if summary['incremental_qaly_mean'] != 0 else "N/A",
            'ICER_Mean': f"€{summary['icer_mean']:,.0f}" if np.isfinite(summary['icer_mean']) else "Dominated/Dominates",
            'ICER_Median': f"€{summary['icer_median']:,.0f}" if np.isfinite(summary['icer_median']) else "Dominated/Dominates",
            'ICER_95CI': f"[€{summary['icer_ci'][0]:,.0f}, €{summary['icer_ci'][1]:,.0f}]" if np.isfinite(summary['icer_ci'][0]) else "N/A",
            'Prob_CE_20k': f"{summary['prob_cost_effective_20k']:.1f}%" if summary['prob_cost_effective_20k'] is not None else "N/A",
            'Prob_CE_50k': f"{summary['prob_cost_effective_50k']:.1f}%" if summary['prob_cost_effective_50k'] is not None else "N/A",
            'Prob_CE_100k': f"{summary['prob_cost_effective_100k']:.1f}%" if summary['prob_cost_effective_100k'] is not None else "N/A",
        }

        summary_data.append(row)

    return pd.DataFrame(summary_data)


def plot_probability_curves_by_horizon(comprehensive_results):
    """
    Plot cost-effectiveness acceptability curves for different time horizons

    Parameters:
    - comprehensive_results: Results from comprehensive_cost_effectiveness_analysis

    Returns:
    - matplotlib figure
    """

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Strict definition (only QALY gains count)
    for horizon_key, horizon_data in comprehensive_results.items():
        prob_df = horizon_data['probability_analysis']
        horizon = horizon_data['summary']['time_horizon']

        ax1.plot(prob_df['threshold'], prob_df['percent_cost_effective_strict'],
                marker='o', linewidth=2, label=f'{horizon} years')

    ax1.set_xlabel('Willingness-to-Pay Threshold (€/QALY)')
    ax1.set_ylabel('Probability Cost-Effective (%)')
    ax1.set_title('Cost-Effectiveness Probability\n(Strict: Requires QALY Gain)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # Add common thresholds
    for thresh in [50000, 100000]:
        ax1.axvline(x=thresh, color='red', linestyle='--', alpha=0.5)
        ax1.text(thresh, 85, f'€{thresh:,}', rotation=90, verticalalignment='bottom')

    # Plot 2: Relaxed definition (includes cost savings)
    for horizon_key, horizon_data in comprehensive_results.items():
        prob_df = horizon_data['probability_analysis']
        horizon = horizon_data['summary']['time_horizon']

        ax2.plot(prob_df['threshold'], prob_df['percent_cost_effective_relaxed'],
                marker='s', linewidth=2, label=f'{horizon} years')

    ax2.set_xlabel('Willingness-to-Pay Threshold (€/QALY)')
    ax2.set_ylabel('Probability Cost-Effective (%)')
    ax2.set_title('Cost-Effectiveness Probability\n(Relaxed: Includes Cost Savings)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    # Add common thresholds
    for thresh in [50000, 100000]:
        ax2.axvline(x=thresh, color='red', linestyle='--', alpha=0.5)
        ax2.text(thresh, 85, f'€{thresh:,}', rotation=90, verticalalignment='bottom')

    plt.tight_layout()
    return fig

