# src/utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

def set_plotting_style():
    """Set consistent plotting style for all visualizations"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18

def plot_yield_curve_forecasts(actual, predicted, dates, maturities, model_name, output_dir):
    """
    Plot actual vs predicted yield curves for selected dates.
    
    Args:
        actual: Array of actual yield curves
        predicted: Array of predicted yield curves
        dates: List of dates to plot
        maturities: List of maturities
        model_name: Name of the model for the title
        output_dir: Directory to save the plots
    """
    set_plotting_style()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    for i, date in enumerate(dates):
        plt.figure(figsize=(10, 6))
        
        # Plot actual yield curve
        plt.plot(maturities, actual[i], 'o-', linewidth=2, label='Actual', color='darkblue')
        
        # Plot predicted yield curve
        plt.plot(maturities, predicted[i], 's--', linewidth=2, label='Predicted', color='crimson')
        
        plt.xlabel('Maturity (months)')
        plt.ylabel('Yield (%)')
        plt.title(f'{model_name} Yield Curve Forecast\n{date}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(output_path / f'yield_curve_{model_name}_{date}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_forecast_errors(errors, horizons, maturities, model_name, output_dir):
    """
    Plot forecast errors across horizons and maturities.
    
    Args:
        errors: Dictionary with horizons as keys and error matrices as values
        horizons: List of forecast horizons
        maturities: List of maturities
        model_name: Name of the model for the title
        output_dir: Directory to save the plots
    """
    set_plotting_style()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Plot errors by horizon
    plt.figure(figsize=(12, 8))
    
    for horizon in horizons:
        horizon_errors = errors[horizon].mean(axis=0)  # Average across time
        plt.plot(maturities, horizon_errors, 'o-', linewidth=2, label=f'h={horizon}')
    
    plt.xlabel('Maturity (months)')
    plt.ylabel('RMSE')
    plt.title(f'{model_name} Forecast Errors by Horizon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path / f'errors_by_horizon_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot errors by maturity
    plt.figure(figsize=(12, 8))
    
    for i, maturity in enumerate(maturities):
        maturity_errors = [errors[horizon][:, i].mean() for horizon in horizons]
        plt.plot(horizons, maturity_errors, 'o-', linewidth=2, label=f'm={maturity}')
    
    plt.xlabel('Forecast Horizon (days)')
    plt.ylabel('RMSE')
    plt.title(f'{model_name} Forecast Errors by Maturity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path / f'errors_by_maturity_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_factors(factors, model_name, output_dir):
    """
    Plot the evolution of estimated factors over time.
    
    Args:
        factors: Array of factors [time, factor]
        model_name: Name of the model for the title
        output_dir: Directory to save the plots
    """
    set_plotting_style()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    plt.figure(figsize=(12, 8))
    
    for i in range(factors.shape[1]):
        plt.plot(factors[:, i], linewidth=2, label=f'Factor {i+1}')
    
    plt.xlabel('Time')
    plt.ylabel('Factor Value')
    plt.title(f'{model_name} Estimated Factors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path / f'factors_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_factor_loadings(loadings, maturities, model_name, output_dir):
    """
    Plot factor loadings across maturities.
    
    Args:
        loadings: Array of factor loadings [maturity, factor]
        maturities: List of maturities
        model_name: Name of the model for the title
        output_dir: Directory to save the plots
    """
    set_plotting_style()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    plt.figure(figsize=(12, 8))
    
    for i in range(loadings.shape[1]):
        plt.plot(maturities, loadings[:, i], 'o-', linewidth=2, label=f'Factor {i+1}')
    
    plt.xlabel('Maturity (months)')
    plt.ylabel('Loading')
    plt.title(f'{model_name} Factor Loadings')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path / f'factor_loadings_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison(comparison_df, metric='rmse', output_dir='results'):
    """
    Plot comparison of multiple models across horizons and maturities.
    
    Args:
        comparison_df: DataFrame with model comparison results
        metric: Metric to plot (default: 'rmse')
        output_dir: Directory to save the plots
    """
    set_plotting_style()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Filter for average results across maturities
    avg_results = comparison_df[comparison_df['maturity'] == 'avg'].copy()
    
    # Plot by horizon
    plt.figure(figsize=(14, 10))
    
    # Create a bar plot
    ax = sns.barplot(x='horizon', y=metric, hue='model', data=avg_results)
    
    plt.xlabel('Forecast Horizon (days)')
    plt.ylabel(f'{metric.upper()} (basis points)')
    plt.title(f'Model Comparison by Forecast Horizon ({metric.upper()})')
    plt.legend(title='Model')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
    
    plt.tight_layout()
    plt.savefig(output_path / f'model_comparison_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot improvement over baseline if data is available
    # First check if we have a Random Walk model to use as baseline
    baseline_models = ['RandomWalk', 'random_walk', 'RW']
    baseline_model = next((m for m in baseline_models if m in avg_results['model'].values), None)
    
    if baseline_model:
        # Create improvement DataFrame for average results
        baseline_avg = avg_results[avg_results['model'] == baseline_model].copy()
        model_improvement_avg = []
        
        for model in avg_results['model'].unique():
            if model != baseline_model:
                for horizon in avg_results['horizon'].unique():
                    baseline_rmse = baseline_avg[baseline_avg['horizon'] == horizon][metric].values
                    model_rmse = avg_results[(avg_results['model'] == model) & 
                                             (avg_results['horizon'] == horizon)][metric].values
                    
                    if len(baseline_rmse) > 0 and len(model_rmse) > 0:
                        improvement = ((baseline_rmse[0] - model_rmse[0]) / baseline_rmse[0]) * 100
                        
                        model_improvement_avg.append({
                            'model': model,
                            'horizon': horizon,
                            'maturity': 'avg',
                            'improvement': improvement,
                            f'{metric}_model': model_rmse[0],
                            f'{metric}_baseline': baseline_rmse[0]
                        })
        
        # Also create improvement DataFrame for each maturity
        # Get numerical maturities
        numerical_maturities = [m for m in comparison_df['maturity'].unique() 
                               if m != 'avg' and pd.notna(m)]
        
        model_improvement_all = []
        model_improvement_all.extend(model_improvement_avg)  # Include average results
        
        for maturity in numerical_maturities:
            baseline_mat = comparison_df[(comparison_df['model'] == baseline_model) & 
                                         (comparison_df['maturity'] == maturity)].copy()
            
            for model in comparison_df['model'].unique():
                if model != baseline_model:
                    for horizon in comparison_df['horizon'].unique():
                        baseline_rmse = baseline_mat[baseline_mat['horizon'] == horizon][metric].values
                        model_rmse = comparison_df[(comparison_df['model'] == model) & 
                                                  (comparison_df['horizon'] == horizon) &
                                                  (comparison_df['maturity'] == maturity)][metric].values
                        
                        if len(baseline_rmse) > 0 and len(model_rmse) > 0:
                            improvement = ((baseline_rmse[0] - model_rmse[0]) / baseline_rmse[0]) * 100
                            
                            model_improvement_all.append({
                                'model': model,
                                'horizon': horizon,
                                'maturity': maturity,
                                'improvement': improvement,
                                f'{metric}_model': model_rmse[0],
                                f'{metric}_baseline': baseline_rmse[0]
                            })
        
        # Save complete improvement DataFrame
        improvement_df = pd.DataFrame(model_improvement_all)
        improvement_df.to_csv(output_path.parent / 'model_improvement.csv', index=False)
        
        # Plot average improvement
        improvement_avg_df = pd.DataFrame(model_improvement_avg)
        improvement_avg_df = improvement_avg_df.sort_values(['horizon', 'improvement'], ascending=[True, False])
        
        plt.figure(figsize=(14, 10))
        ax = sns.barplot(x='horizon', y='improvement', hue='model', data=improvement_avg_df)
        
        plt.xlabel('Forecast Horizon (days)')
        plt.ylabel('Improvement over Baseline (%)')
        plt.title(f'Model Improvement over {baseline_model} (Average across maturities)')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.legend(title='Model')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f%%', padding=3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'model_improvement_avg.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot improvement by maturity for each horizon
        for horizon in improvement_df['horizon'].unique():
            # Get top numerical maturities to plot (to avoid too crowded plot)
            selected_maturities = sorted(numerical_maturities)[:7]  # Select first 7 maturities
            
            horizon_imp = improvement_df[(improvement_df['horizon'] == horizon) & 
                                         (improvement_df['maturity'].isin(selected_maturities))]
            
            if len(horizon_imp) > 0:
                plt.figure(figsize=(14, 8))
                g = sns.catplot(
                    data=horizon_imp, 
                    x='maturity', 
                    y='improvement',
                    hue='model',
                    kind='bar',
                    height=6,
                    aspect=1.8
                )
                
                plt.xlabel('Maturity (months)')
                plt.ylabel('Improvement over Baseline (%)')
                plt.title(f'Model Improvement over {baseline_model} by Maturity ({horizon}-day horizon)')
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_path / f'model_improvement_h{horizon}_by_maturity.png', dpi=300, bbox_inches='tight')
                plt.close()

def load_forecast_errors(models_dirs, horizons, maturities):
    """
    Load forecast errors from result files for multiple models.
    
    Args:
        models_dirs: Dictionary mapping model names to their result directories
        horizons: List of forecast horizons to consider
        maturities: List of maturities to visualize
        
    Returns:
        Dictionary with model names as keys and nested dictionaries for horizons and maturities
    """
    all_errors = {}
    
    for model_name, model_dir in models_dirs.items():
        model_errors = {}
        
        for horizon in horizons:
            horizon_errors = {}
            error_file = Path(model_dir) / f'errors_h{horizon}.csv'
            
            if error_file.exists():
                try:
                    # Try to load errors from CSV file
                    df = pd.read_csv(error_file)
                    
                    # Process each maturity
                    for maturity in maturities:
                        if f'mat_{maturity}' in df.columns:
                            # Extract absolute errors for this maturity
                            errors = df[f'mat_{maturity}'].abs().values
                            horizon_errors[maturity] = errors
                        elif str(maturity) in df.columns:
                            # Alternative column naming
                            errors = df[str(maturity)].abs().values
                            horizon_errors[maturity] = errors
                except Exception as e:
                    print(f"Error loading data for {model_name}, horizon {horizon}: {e}")
            
            # Try numpy format if CSV doesn't exist
            if not horizon_errors:
                error_file = Path(model_dir) / f'errors_h{horizon}.npy'
                if error_file.exists():
                    try:
                        errors_array = np.load(error_file)
                        
                        # Assuming the array has maturities as columns
                        for i, maturity in enumerate(maturities):
                            if i < errors_array.shape[1]:
                                horizon_errors[maturity] = np.abs(errors_array[:, i])
                    except Exception as e:
                        print(f"Error loading numpy data for {model_name}, horizon {horizon}: {e}")
            
            if horizon_errors:
                model_errors[horizon] = horizon_errors
        
        if model_errors:
            all_errors[model_name] = model_errors
    
    return all_errors

def plot_error_boxplots(errors, models, horizons, maturities, output_dir):
    """
    Create boxplots of forecast errors for different models, horizons, and maturities.
    
    Args:
        errors: Dictionary with model names as keys and error data as values
        models: List of model names
        horizons: List of forecast horizons
        maturities: List of maturities
        output_dir: Directory to save the plots
    """
    set_plotting_style()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Plot boxplots by horizon for each maturity
    for i, maturity in enumerate(maturities):
        plt.figure(figsize=(14, 8))
        
        data_to_plot = []
        labels = []
        
        for model in models:
            if model in errors:
                for horizon in horizons:
                    if horizon in errors[model]:
                        # Extract errors for this maturity
                        maturity_errors = errors[model][horizon][:, i]
                        data_to_plot.append(maturity_errors)
                        labels.append(f"{model} (h={horizon})")
        
        if data_to_plot:
            plt.boxplot(data_to_plot, labels=labels, showfliers=False)
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Model and Horizon')
            plt.ylabel('Forecast Error (basis points)')
            plt.title(f'Forecast Error Distribution - Maturity {maturity} months')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(output_path / f'error_boxplot_m{maturity}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot boxplots by model for each horizon
    for horizon in horizons:
        plt.figure(figsize=(14, 8))
        
        data_to_plot = []
        labels = []
        
        for model in models:
            if model in errors and horizon in errors[model]:
                # Average across maturities
                avg_errors = errors[model][horizon].mean(axis=1)
                data_to_plot.append(avg_errors)
                labels.append(model)
        
        if data_to_plot:
            plt.boxplot(data_to_plot, labels=labels, showfliers=False)
            plt.xlabel('Model')
            plt.ylabel('Average Forecast Error (basis points)')
            plt.title(f'Average Forecast Error Distribution - Horizon {horizon} days')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(output_path / f'error_boxplot_h{horizon}_avg.png', dpi=300, bbox_inches='tight')
            plt.close()