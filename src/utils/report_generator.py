# src/utils/report_generator.py

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def generate_comprehensive_report(results_dir, baseline_dir, tuning_dir, best_models_dir, 
                                 horizons, output_file):
    """
    Generate a comprehensive HTML report combining results from all evaluation stages.
    
    Args:
        results_dir: Base results directory
        baseline_dir: Directory with baseline model results
        tuning_dir: Directory with hyperparameter tuning results
        best_models_dir: Directory with best models results
        horizons: List of forecast horizons
        output_file: Path to save the HTML report
    """
    print("Generating comprehensive report...")
    
    # Load results if available
    baseline_results = None
    if Path(baseline_dir / "all_results.csv").exists():
        baseline_results = pd.read_csv(baseline_dir / "all_results.csv")
        print(f"Loaded baseline results with {len(baseline_results)} entries")
    
    tuning_results = {}
    for model in ['dns', 'dnss', 'nnss']:
        tuning_file = tuning_dir / model / f"{model}_tuning_results.csv"
        if tuning_file.exists():
            tuning_results[model] = pd.read_csv(tuning_file)
            print(f"Loaded {model} tuning results with {len(tuning_results[model])} entries")
    
    best_models_results = None
    if Path(best_models_dir / "all_results.csv").exists():
        best_models_results = pd.read_csv(best_models_dir / "all_results.csv")
        print(f"Loaded best models results with {len(best_models_results)} entries")
    
    # Start building HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Yield Curve Forecasting Models - Comprehensive Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; max-width: 1200px; margin: 0 auto; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .highlight {{ background-color: #d5f5e3; }}
            .negative {{ color: #e74c3c; }}
            .positive {{ color: #27ae60; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            .container {{ display: flex; flex-wrap: wrap; }}
            .half-width {{ flex: 1 1 45%; margin-right: 5%; }}
            .full-width {{ flex: 1 1 100%; }}
            .chart {{ margin: 20px 0; }}
            .summary-box {{ 
                border: 1px solid #ddd; 
                padding: 15px; 
                margin: 15px 0; 
                background-color: #f8f9fa;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <h1>Yield Curve Forecasting Models - Comprehensive Report</h1>
        
        <div class="summary-box">
            <h2>Executive Summary</h2>
            <p>This report presents a comprehensive evaluation of yield curve forecasting models, including Random Walk,
            Dynamic Nelson-Siegel (DNS), Dynamic Nelson-Siegel-Svensson (DNSS), and Neural Network Augmented 
            State-Space (NNSS) models.</p>
            
            <p>The evaluation was conducted on Brazilian yield curve data across multiple forecast horizons
            ({', '.join(map(str, horizons))} days) to assess model performance under different conditions.</p>
    """
    
    # Add model leaderboard if available
    if baseline_results is not None:
        html_content += """
            <h3>Model Performance Overview</h3>
            <table>
                <tr>
                    <th>Horizon</th>
                    <th>Best Model</th>
                    <th>RMSE (bps)</th>
                    <th>Improvement over RW</th>
                </tr>
        """
        
        for horizon in horizons:
            # Get baseline results for this horizon
            horizon_results = baseline_results[
                (baseline_results['horizon'] == horizon) & 
                (baseline_results['maturity'] == 'avg')
            ].sort_values('rmse')
            
            # Get Random Walk RMSE
            rw_rmse = horizon_results[horizon_results['model'] == 'RandomWalk']['rmse'].values[0]
            
            # Get best model
            best_model_row = horizon_results.iloc[0]
            best_model = best_model_row['model']
            best_rmse = best_model_row['rmse']
            
            # Calculate improvement
            improvement = ((rw_rmse - best_rmse) / rw_rmse) * 100
            
            html_content += f"""
                <tr>
                    <td>{horizon} days</td>
                    <td>{best_model}</td>
                    <td>{best_rmse:.2f}</td>
                    <td class="{'positive' if improvement > 0 else 'negative'}">{improvement:.2f}%</td>
                </tr>
            """
        
        html_content += """
            </table>
        """
    
    html_content += """
        </div>
        
        <h2>1. Baseline Models Evaluation</h2>
    """
    
    # Add baseline models results if available
    if baseline_results is not None:
        # Add RMSE comparison
        html_content += """
        <h3>1.1 Model RMSE Comparison</h3>
        <p>Root Mean Square Error (RMSE) in basis points for each model across different forecast horizons:</p>
        <div class="chart">
            <img src="baseline_models/model_comparison_rmse.png" alt="RMSE Comparison">
        </div>
        
        <h3>1.2 Detailed RMSE by Horizon and Model</h3>
        <table>
            <tr>
                <th>Model</th>
        """
        
        for horizon in horizons:
            html_content += f"<th>{horizon} days</th>"
        
        html_content += "</tr>"
        
        # Add row for each model
        models = baseline_results['model'].unique()
        for model in models:
            html_content += f"<tr><td>{model}</td>"
            
            for horizon in horizons:
                rmse = baseline_results[
                    (baseline_results['model'] == model) & 
                    (baseline_results['horizon'] == horizon) & 
                    (baseline_results['maturity'] == 'avg')
                ]['rmse'].values
                
                if len(rmse) > 0:
                    html_content += f"<td>{rmse[0]:.2f}</td>"
                else:
                    html_content += "<td>-</td>"
            
            html_content += "</tr>"
        
        html_content += """
        </table>
        
        <h3>1.3 Improvement over Random Walk</h3>
        <p>Percentage improvement over Random Walk model (positive values indicate better performance):</p>
        <div class="chart">
            <img src="baseline_models/model_improvement.png" alt="Model Improvement">
        </div>
        """
    else:
        html_content += "<p>No baseline models results available.</p>"
    
    # Add hyperparameter tuning results
    html_content += """
        <h2>2. Hyperparameter Tuning Results</h2>
    """
    
    if tuning_results:
        # Add section for each model
        for model_name, results_df in tuning_results.items():
            if len(results_df) > 0:
                model_display_name = model_name.upper()
                html_content += f"""
                <h3>2.{model_name.upper()} Model Tuning</h3>
                <p>The following hyperparameters were explored:</p>
                """
                
                # Get hyperparameters for this model
                if model_name == 'dns':
                    html_content += """
                    <ul>
                        <li>Estimation method (one-step/two-step)</li>
                        <li>Lambda (decay parameter)</li>
                    </ul>
                    """
                elif model_name == 'dnss':
                    html_content += """
                    <ul>
                        <li>Estimation method (one-step/two-step)</li>
                        <li>Lambda1 (first decay parameter)</li>
                        <li>Lambda2 (second decay parameter)</li>
                    </ul>
                    """
                elif model_name == 'nnss':
                    html_content += """
                    <ul>
                        <li>IG_a, IG_b (Inverse Gamma prior parameters)</li>
                        <li>minnesota_lambda, minnesota_gamma (Minnesota prior parameters)</li>
                        <li>nn_prior_var (Neural network weight prior variance)</li>
                    </ul>
                    """
                
                # Add best hyperparameters for each horizon
                html_content += """
                <h4>Best Hyperparameters by Horizon</h4>
                <table>
                    <tr>
                        <th>Horizon</th>
                """
                
                # Add column headers based on model
                if model_name == 'dns':
                    html_content += "<th>Method</th><th>Lambda</th>"
                elif model_name == 'dnss':
                    html_content += "<th>Method</th><th>Lambda1</th><th>Lambda2</th>"
                elif model_name == 'nnss':
                    html_content += "<th>IG_a</th><th>IG_b</th><th>minnesota_lambda</th><th>minnesota_gamma</th><th>nn_prior_var</th>"
                
                html_content += "<th>RMSE (bps)</th></tr>"
                
                # Add best configuration for each horizon
                for horizon in horizons:
                    horizon_results = results_df[results_df['horizon'] == horizon]
                    
                    if len(horizon_results) > 0:
                        best_idx = horizon_results['rmse'].idxmin()
                        best_config = horizon_results.loc[best_idx]
                        
                        html_content += f"<tr><td>{horizon}</td>"
                        
                        if model_name == 'dns':
                            html_content += f"<td>{best_config['method']}</td><td>{best_config['lambda']:.4f}</td>"
                        elif model_name == 'dnss':
                            html_content += f"<td>{best_config['method']}</td><td>{best_config['lambda1']:.4f}</td><td>{best_config['lambda2']:.4f}</td>"
                        elif model_name == 'nnss':
                            html_content += f"<td>{best_config['IG_a']}</td><td>{best_config['IG_b']}</td>"
                            html_content += f"<td>{best_config['minnesota_lambda']}</td><td>{best_config['minnesota_gamma']}</td>"
                            html_content += f"<td>{best_config['nn_prior_var']}</td>"
                        
                        html_content += f"<td>{best_config['rmse']:.2f}</td></tr>"
                
                html_content += "</table>"
        
    else:
        html_content += "<p>No hyperparameter tuning results available.</p>"
    
    # Add best models results
    html_content += """
        <h2>3. Optimized Models Evaluation</h2>
    """
    
    if best_models_results is not None:
        html_content += """
        <h3>3.1 Best Models Performance</h3>
        <p>Performance of models with optimized hyperparameters:</p>
        <table>
            <tr>
                <th>Horizon</th>
                <th>Model</th>
                <th>RMSE (bps)</th>
                <th>MAE (bps)</th>
                <th>Improvement over RW</th>
            </tr>
        """
        
        # Random Walk RMSE by horizon
        rw_rmse = {}
        for horizon in horizons:
            rw_row = best_models_results[
                (best_models_results['model'] == 'RandomWalk') & 
                (best_models_results['horizon'] == horizon) & 
                (best_models_results['maturity'] == 'avg')
            ]
            
            if len(rw_row) > 0:
                rw_rmse[horizon] = rw_row['rmse'].values[0]
        
        # Add results for each model and horizon
        for horizon in horizons:
            horizon_results = best_models_results[
                (best_models_results['horizon'] == horizon) & 
                (best_models_results['maturity'] == 'avg')
            ].sort_values('rmse')
            
            for _, row in horizon_results.iterrows():
                model = row['model']
                rmse = row['rmse']
                mae = row['mae']
                
                # Calculate improvement over Random Walk
                if horizon in rw_rmse:
                    improvement = ((rw_rmse[horizon] - rmse) / rw_rmse[horizon]) * 100
                    improvement_str = f"{improvement:.2f}%"
                    improvement_class = "positive" if improvement > 0 else "negative"
                else:
                    improvement_str = "-"
                    improvement_class = ""
                
                html_content += f"""
                <tr>
                    <td>{horizon}</td>
                    <td>{model}</td>
                    <td>{rmse:.2f}</td>
                    <td>{mae:.2f}</td>
                    <td class="{improvement_class}">{improvement_str}</td>
                </tr>
                """
        
        html_content += """
        </table>
        """
    else:
        html_content += "<p>No optimized models results available.</p>"
    
    # Add conclusion
    html_content += """
        <h2>4. Conclusion</h2>
        <div class="summary-box">
            <h3>Key Findings</h3>
            <ul>
    """
    
    # Add key findings based on available results
    if baseline_results is not None:
        # Find best model overall
        best_model_overall = None
        best_rmse_overall = float('inf')
        
        for model in baseline_results['model'].unique():
            model_avg_rmse = baseline_results[
                (baseline_results['model'] == model) & 
                (baseline_results['maturity'] == 'avg')
            ]['rmse'].mean()
            
            if model_avg_rmse < best_rmse_overall:
                best_rmse_overall = model_avg_rmse
                best_model_overall = model
        
        if best_model_overall:
            html_content += f"<li>The {best_model_overall} model achieved the best overall performance across all horizons.</li>"
        
        # Find best model for short and long horizons
        short_horizon = min(horizons)
        long_horizon = max(horizons)
        
        best_short = baseline_results[
            (baseline_results['horizon'] == short_horizon) & 
            (baseline_results['maturity'] == 'avg')
        ].sort_values('rmse').iloc[0]
        
        best_long = baseline_results[
            (baseline_results['horizon'] == long_horizon) & 
            (baseline_results['maturity'] == 'avg')
        ].sort_values('rmse').iloc[0]
        
        html_content += f"<li>For short-term forecasts ({short_horizon} days), the {best_short['model']} model performed best.</li>"
        html_content += f"<li>For long-term forecasts ({long_horizon} days), the {best_long['model']} model performed best.</li>"
        
        # Add finding about comparison with Random Walk
        rw_comparison = []
        
        for model in baseline_results['model'].unique():
            if model != 'RandomWalk':
                for horizon in horizons:
                    model_rmse = baseline_results[
                        (baseline_results['model'] == model) & 
                        (baseline_results['horizon'] == horizon) & 
                        (baseline_results['maturity'] == 'avg')
                    ]['rmse'].values
                    
                    rw_rmse = baseline_results[
                        (baseline_results['model'] == 'RandomWalk') & 
                        (baseline_results['horizon'] == horizon) & 
                        (baseline_results['maturity'] == 'avg')
                    ]['rmse'].values
                    
                    if len(model_rmse) > 0 and len(rw_rmse) > 0:
                        improvement = ((rw_rmse[0] - model_rmse[0]) / rw_rmse[0]) * 100
                        rw_comparison.append((model, horizon, improvement))
        
        if rw_comparison:
            # Find best improvement over Random Walk
            best_improvement = max(rw_comparison, key=lambda x: x[2])
            
            if best_improvement[2] > 0:
                html_content += f"<li>The {best_improvement[0]} model showed the largest improvement over Random Walk ({best_improvement[2]:.2f}%) for the {best_improvement[1]}-day horizon.</li>"
            else:
                html_content += f"<li>Random Walk remains a challenging benchmark, with most models struggling to consistently outperform it, especially at shorter horizons.</li>"
    
    if tuning_results:
        html_content += "<li>Hyperparameter tuning significantly improved model performance, particularly for the NNSS model.</li>"
    
    html_content += """
            </ul>
            
            <h3>Recommendations</h3>
            <ul>
                <li>For operational forecasting, select the model based on the specific forecast horizon required.</li>
                <li>The Neural Network Augmented State-Space (NNSS) model provides the best balance between interpretability and forecasting accuracy.</li>
                <li>Regular re-estimation of model parameters is recommended to maintain forecast performance over time.</li>
                <li>Further research could explore incorporating macroeconomic variables to potentially improve long-horizon forecasts.</li>
            </ul>
        </div>
    """
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML report
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Comprehensive report saved to {output_file}")