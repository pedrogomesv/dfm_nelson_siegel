import os
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from src.utils.visualization import (
    set_plotting_style,
    plot_model_comparison,
    load_forecast_errors,
    plot_error_boxplots,
)
from src.models.baselines.random_walk import RandomWalkModel
from src.models.baselines.dns import DynamicNelsonSiegelModel
from src.models.baselines.dnss import DynamicNelsonSiegelSvenssonModel
from src.models.baselines.nnss import NNSSModel
from src.models.resnet.resnet import NNSSResNetModel


def load_yield_data(file_path):
    """
    Carrega os dados da curva de juros a partir de um arquivo.

    Args:
        file_path: insira o path aqui

    Returns:
        DataFrame com os dados da curva de juros
    """
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    return df


def run_baseline_models(
    data_path,
    output_dir,
    train_test_split=0.8,
    horizons=[5, 20, 60, 120],
    use_gpu=False,
    nnss_epochs=500,
    nnss_patience=10,
    validation_split=0.15,
):
    """
    Executa todos os modelos baseline e compara o desempenho.
    Inclui parada antecipada (early stopping) para modelos baseados em NNSS usando uma divisão de validação.

    Args:
        data_path: insira o path aqui
        output_dir: insira o path aqui
        train_test_split: Proporção dos dados a serem usados para treino
        horizons: Lista de horizontes de previsão
        use_gpu: Se deve usar GPU para o modelo NNSS
        nnss_epochs: Número máximo de épocas para o treinamento do NNSS
        nnss_patience: Paciência para parada antecipada no treinamento do NNSS
        validation_split: Proporção dos dados de treino a serem usados para validação no NNSS

    Returns:
        DataFrame com os resultados comparativos
    """
    device = "cuda" if use_gpu else "cpu"

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    yield_data = load_yield_data(data_path)

    maturities = [int(col) for col in yield_data.columns]
    dates = yield_data.index

    split_idx = int(len(yield_data) * train_test_split)
    initial_train_data = yield_data.iloc[:split_idx]
    test_data = yield_data.iloc[split_idx:]

    train_subset, val_subset = sklearn_train_test_split(
        initial_train_data,
        test_size=validation_split,
        shuffle=False,
    )

    print(f"Data loaded: {len(yield_data)} observations")
    print(f"Initial Train data: {len(initial_train_data)} observations")
    print(f"  -> Train Subset: {len(train_subset)} observations")
    print(f"  -> Validation Subset: {len(val_subset)} observations")
    print(f"Test data: {len(test_data)} observations")
    print(f"Maturities: {maturities}")

    all_results = {}

    models = {
        "RandomWalk": RandomWalkModel(maturities),
        "DynamicNelsonSiegelTwoStep": DynamicNelsonSiegelModel(
            maturities, method="two-step"
        ),
        "DynamicNelsonSiegelOneStep": DynamicNelsonSiegelModel(
            maturities, method="one-step"
        ),
        "DynamicNelsonSiegelSvenssonTwoStep": DynamicNelsonSiegelSvenssonModel(
            maturities, method="two-step"
        ),
        "DynamicNelsonSiegelSvenssonOneStep": DynamicNelsonSiegelSvenssonModel(
            maturities, method="one-step"
        ),
        "NNSS": NNSSModel(maturities, n_factors=4, device=device),
        "NNSS_ResNet": NNSSResNetModel(
            maturities, n_factors=4, n_blocks=3, device=device
        ),
    }

    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Running {model_name}")
        print(f"{'='*50}")

        if model_name in ["NNSS", "NNSS_ResNet"]:
            checkpoint_dir = output_path / model_name / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"  Training with validation split, epochs={nnss_epochs}, patience={nnss_patience}"
            )
            model.fit(
                train_data=train_subset,
                val_data=val_subset,
                epochs=nnss_epochs,
                patience=nnss_patience,
                checkpoint_dir=checkpoint_dir,
            )
        else:
            model.fit(initial_train_data)

        model_output_dir = output_path / model_name
        model.save(model_output_dir)

        results = model.evaluate(test_data, horizons, output_dir=model_output_dir)
        all_results[model_name] = results

        print(f"\nResults for {model_name}:")
        for horizon in horizons:
            avg_rmse = results[results["horizon"] == horizon][
                results["maturity"] == "avg"
            ]["rmse"].values[0]
            print(f"  Horizon {horizon}: RMSE = {avg_rmse:.2f} bps")

    comparison = pd.concat([df.assign(model=name) for name, df in all_results.items()])

    comparison.to_csv(output_path / "all_results.csv", index=False)

    plot_model_comparison(comparison, metric="rmse", output_dir=output_dir)

    baseline_results = all_results["RandomWalk"]
    model_comparison = {}

    for model_name, results_df in all_results.items():
        if model_name == "RandomWalk":
            continue

        merged = pd.merge(
            results_df,
            baseline_results[["horizon", "maturity", "rmse"]],
            on=["horizon", "maturity"],
            suffixes=("", "_baseline"),
        )

        merged["improvement"] = (
            (merged["rmse_baseline"] - merged["rmse"]) / merged["rmse_baseline"]
        ) * 100

        merged["model"] = model_name

        model_comparison[model_name] = merged

    improvement_df = pd.concat(model_comparison.values())
    improvement_df.to_csv(output_path / "model_improvement.csv", index=False)

    leaderboard = []

    for horizon in horizons:
        horizon_results = comparison[
            (comparison["horizon"] == horizon) & (comparison["maturity"] == "avg")
        ].sort_values("rmse")

        for i, (_, row) in enumerate(horizon_results.iterrows()):
            leaderboard.append(
                {
                    "horizon": horizon,
                    "rank": i + 1,
                    "model": row["model"],
                    "rmse": row["rmse"],
                    "mae": row["mae"],
                }
            )

    leaderboard_df = pd.DataFrame(leaderboard)
    leaderboard_df.to_csv(output_path / "leaderboard.csv", index=False)

    print("\nModel Leaderboard:")
    for horizon in horizons:
        print(f"\nHorizon {horizon}:")
        horizon_board = leaderboard_df[leaderboard_df["horizon"] == horizon]
        for _, row in horizon_board.iterrows():
            print(
                f"  Rank {int(row['rank'])}: {row['model']} (RMSE = {row['rmse']:.2f} bps)"
            )

    generate_summary_report(
        all_results,
        improvement_df,
        leaderboard_df,
        initial_train_data,
        test_data,
        output_dir,
    )

    return comparison


def generate_summary_report(
    all_results,
    improvement_df,
    leaderboard_df,
    train_data,
    test_data,
    output_dir,
    vis_dir=None,
):
    """
    Gera um relatório resumido da comparação dos modelos.

    Args:
        all_results: Dicionário com os resultados dos modelos
        improvement_df: DataFrame com a melhoria sobre o baseline
        leaderboard_df: DataFrame com o ranking dos modelos
        train_data: Dados de treinamento
        test_data: Dados de teste
        output_dir: insira o path aqui
        vis_dir: insira o path aqui
    """
    output_path = Path(output_dir)

    if vis_dir is None:
        vis_dir = output_path / "visualizations"
    else:
        vis_dir = Path(vis_dir)

    vis_path_rel = (
        vis_dir.relative_to(output_path)
        if vis_dir.is_relative_to(output_path)
        else vis_dir
    )

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Yield Curve Forecasting Model Comparison</title>
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
            .full-width {{ flex: 1 1 100%; margin-bottom: 30px; }}
        </style>
    </head>
    <body>
        <h1>Yield Curve Forecasting Model Comparison</h1>
        
        <h2>Dataset Overview</h2>
        <p>
            <strong>Train period:</strong> {train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')} ({len(train_data)} observations)<br>
            <strong>Test period:</strong> {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')} ({len(test_data)} observations)<br>
            <strong>Maturities:</strong> {', '.join([str(col) for col in train_data.columns])} months
        </p>
    """

    rmse_plot = vis_dir / "model_comparison_rmse.png"
    improvement_plot = vis_dir / "model_improvement.png"

    if rmse_plot.exists() or improvement_plot.exists():
        html_content += """
        <h2>Model Comparison Visualizations</h2>
        <div class="container">
        """

        if rmse_plot.exists():
            vis_path = str(vis_path_rel / "model_comparison_rmse.png")
            html_content += f"""
            <div class="half-width">
                <h3>RMSE by Horizon</h3>
                <img src="{vis_path}" alt="RMSE Comparison">
            </div>
            """

        if improvement_plot.exists():
            vis_path = str(vis_path_rel / "model_improvement.png")
            html_content += f"""
            <div class="half-width">
                <h3>Improvement over Random Walk</h3>
                <img src="{vis_path}" alt="Model Improvement">
            </div>
            """

        html_content += "</div>"

    boxplot_files = list(vis_dir.glob("boxplot_*.png"))
    if boxplot_files:
        html_content += """
        <h2>Forecast Error Analysis</h2>
        """

        comparison_boxplots = sorted(
            [f for f in boxplot_files if "comparison" in f.name]
        )
        if comparison_boxplots:
            html_content += """
            <h3>Model Comparison by Horizon</h3>
            <p>These boxplots show the distribution of absolute forecast errors for different models and maturities.</p>
            """

            for boxplot in comparison_boxplots:
                vis_path = str(vis_path_rel / boxplot.name)
                horizon = boxplot.name.split("_h")[1].split("_")[0]
                html_content += f"""
                <div class="full-width">
                    <h4>{horizon}-day Horizon</h4>
                    <img src="{vis_path}" alt="Boxplot for {horizon}-day horizon">
                </div>
                """

        maturity_boxplots = sorted(
            [f for f in boxplot_files if "by_maturity" in f.name]
        )
        if maturity_boxplots:
            html_content += """
            <h3>Errors by Maturity</h3>
            <p>These boxplots show the forecast errors for each maturity across different models.</p>
            """

            for boxplot in maturity_boxplots:
                vis_path = str(vis_path_rel / boxplot.name)
                horizon = boxplot.name.split("_h")[1].split("_")[0]
                html_content += f"""
                <div class="full-width">
                    <h4>{horizon}-day Horizon</h4>
                    <img src="{vis_path}" alt="Boxplot by maturity for {horizon}-day horizon">
                </div>
                """

    html_content += """
        <h2>Model Leaderboard</h2>
    """

    for horizon in leaderboard_df["horizon"].unique():
        horizon_board = leaderboard_df[
            leaderboard_df["horizon"] == horizon
        ].sort_values("rank")

        html_content += f"""
        <h3>Horizon: {horizon} days</h3>
        <table>
            <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>RMSE (bps)</th>
                <th>MAE (bps)</th>
            </tr>
        """

        for _, row in horizon_board.iterrows():
            highlight = ' class="highlight"' if row.get("rank", 0) == 1 else ""
            html_content += f"""
            <tr{highlight}>
                <td>{int(row.get('rank', 0))}</td>
                <td>{row.get('model', 'Unknown')}</td>
                <td>{row.get('rmse', 0):.2f}</td>
                <td>{row.get('mae', 0):.2f}</td>
            </tr>
            """

        html_content += "</table>"

    if improvement_df is not None:
        html_content += """
        <h2>Improvement over Random Walk</h2>
        <p>Positive values indicate better performance than Random Walk.</p>
    """

        avg_improvement = improvement_df[improvement_df["maturity"] == "avg"].copy()

        for horizon in avg_improvement["horizon"].unique():
            horizon_imp = avg_improvement[
                avg_improvement["horizon"] == horizon
            ].sort_values("improvement", ascending=False)

            html_content += f"""
            <h3>Horizon: {horizon} days</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>RMSE (bps)</th>
                    <th>RW RMSE (bps)</th>
                    <th>Improvement (%)</th>
                </tr>
            """

            for _, row in horizon_imp.iterrows():
                imp_class = "positive" if row.get("improvement", 0) > 0 else "negative"
                html_content += f"""
                <tr>
                    <td>{row.get('model', 'Unknown')}</td>
                    <td>{row.get('rmse', 0):.2f}</td>
                    <td>{row.get('rmse_baseline', 0):.2f}</td>
                    <td class="{imp_class}">{row.get('improvement', 0):.2f}%</td>
                </tr>
                """

            html_content += "</table>"

    html_content += """
        <h2>Detailed Model Performance</h2>
    """

    for model_name in [m for m in all_results.keys() if m != "RandomWalk"]:
        model_results = all_results[model_name]

        html_content += f"""
        <h3>{model_name}</h3>
        """

        factor_plot = None
        loading_plot = None

        for img_file in vis_dir.glob(f"{model_name}_*_factors.png"):
            factor_plot = img_file

        for img_file in vis_dir.glob(f"{model_name}_*_factor_loadings.png"):
            loading_plot = img_file

        if factor_plot and loading_plot:
            factor_vis_path = str(vis_path_rel / factor_plot.name)
            loading_vis_path = str(vis_path_rel / loading_plot.name)

            html_content += f"""
            <div class="container">
                <div class="half-width">
                    <img src="{factor_vis_path}" alt="{model_name} Factors">
                </div>
                
                <div class="half-width">
                    <img src="{loading_vis_path}" alt="{model_name} Factor Loadings">
                </div>
            </div>
            """

        html_content += f"""
        <h4>Performance by Maturity (RMSE in bps)</h4>
        <table>
            <tr>
                <th>Maturity</th>
        """

        for horizon in model_results["horizon"].unique():
            if horizon in [5, 20, 60, 120]:
                html_content += f"<th>{horizon} days</th>"

        html_content += "</tr>"

        mat_results = model_results[model_results["maturity"] != "avg"].copy()
        mat_results["maturity"] = pd.to_numeric(mat_results["maturity"])
        mat_results = mat_results.sort_values("maturity")

        for maturity in mat_results["maturity"].unique():
            html_content += f"""
            <tr>
                <td>{maturity}</td>
            """

            for horizon in [5, 20, 60, 120]:
                filtered_results = mat_results[
                    (mat_results["maturity"] == maturity)
                    & (mat_results["horizon"] == horizon)
                ]

                if len(filtered_results) > 0 and "rmse" in filtered_results.columns:
                    val = filtered_results["rmse"].values[0]
                    html_content += f"<td>{val:.2f}</td>"
                else:
                    html_content += "<td>-</td>"

            html_content += "</tr>"

        html_content += "</table>"

    if "RandomWalk" in all_results:
        model_name = "RandomWalk"
        model_results = all_results[model_name]

        html_content += f"""
        <h3>{model_name}</h3>
        <p>Random Walk model does not have factor loadings or factors.</p>
        <h4>Performance by Maturity (RMSE in bps)</h4>
        <table>
            <tr>
                <th>Maturity</th>
        """

        for horizon in model_results["horizon"].unique():
            if horizon in [5, 20, 60, 120]:
                html_content += f"<th>{horizon} days</th>"

        html_content += "</tr>"

        mat_results = model_results[model_results["maturity"] != "avg"].copy()
        mat_results["maturity"] = pd.to_numeric(mat_results["maturity"])
        mat_results = mat_results.sort_values("maturity")

        for maturity in mat_results["maturity"].unique():
            html_content += f"""
            <tr>
                <td>{maturity}</td>
            """

            for horizon in [5, 20, 60, 120]:
                filtered_results = mat_results[
                    (mat_results["maturity"] == maturity)
                    & (mat_results["horizon"] == horizon)
                ]

                if len(filtered_results) > 0 and "rmse" in filtered_results.columns:
                    val = filtered_results["rmse"].values[0]
                    html_content += f"<td>{val:.2f}</td>"
                else:
                    html_content += "<td>-</td>"

            html_content += "</tr>"

        html_content += "</table>"

    html_content += """
    </body>
    </html>
    """

    with open(output_path / "summary_report.html", "w") as f:
        f.write(html_content)

    print(f"\nSummary report saved to {output_path / 'summary_report.html'}")
