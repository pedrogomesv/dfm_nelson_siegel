import os
import sys
from pathlib import Path
import time
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sklearn_train_test_split

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from src.utils.visualization import set_plotting_style, plot_model_comparison
from src.run_baseline_models import run_baseline_models
from src.tune_hyperparameters import run_hyperparameter_tuning
from src.run_best_models import run_best_models


def main(args):
    """
    Executa o fluxo de trabalho completo de avaliação dos modelos de linha de base.

    Isso inclui:
    1. Execução de todos os modelos de linha de base com configurações padrão
    2. Ajuste de hiperparâmetros (opcional)
    3. Execução de modelos com os melhores hiperparâmetros (opcional)
    4. Geração de relatórios abrangentes
    """
    start_time = time.time()

    results_dir = Path(args.output_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    baseline_dir = results_dir / "baseline_models"
    tuning_dir = results_dir / "hyperparameter_tuning"
    best_models_dir = results_dir / "best_models"

    data = pd.read_csv(args.data_path)
    maturities = np.array([col for col in data.columns if col.startswith("M")])

    print(f"\n{'='*80}")
    print(f"Starting Yield Curve Forecasting Baseline Model Evaluation")
    print(f"{'='*80}")

    print(f"\nData path: {args.data_path}")
    print(f"Results will be saved to: {results_dir}")
    print(f"Horizons to evaluate: {args.horizons}")
    print(f"Using GPU: {args.use_gpu}")

    if not Path(args.data_path).exists():
        print(f"ERROR: Data file not found at {args.data_path}")
        print("Please provide the correct path to the yield curve data.")
        return

    if not args.skip_baseline:
        print(f"\n{'-'*80}")
        print(f"STAGE 1: Running baseline models with default configurations")
        print(f"{'-'*80}")

        stage1_start = time.time()

        baseline_results = run_baseline_models(
            args.data_path,
            baseline_dir,
            train_test_split=args.train_test_split,
            horizons=args.horizons,
            use_gpu=args.use_gpu,
            nnss_epochs=args.nnss_epochs,
            nnss_patience=args.nnss_patience,
            validation_split=args.validation_split,
        )

        stage1_time = time.time() - stage1_start
        print(f"\nStage 1 completed in {stage1_time:.2f} seconds")
    else:
        print(f"\nSkipping Stage 1 (baseline models) as requested")

    if not args.skip_tuning:
        print(f"\n{'-'*80}")
        print(f"STAGE 2: Hyperparameter tuning")
        print(f"{'-'*80}")

        stage2_start = time.time()

        best_params = run_hyperparameter_tuning(
            args.data_path,
            tuning_dir,
            train_val_test_split=[0.7, 0.15, 0.15],
            horizons=args.horizons,
            tune_models=args.tune_models,
            use_gpu=args.use_gpu,
        )

        stage2_time = time.time() - stage2_start
        print(f"\nStage 2 completed in {stage2_time:.2f} seconds")
    else:
        print(f"\nSkipping Stage 2 (hyperparameter tuning) as requested")

    if not args.skip_best_models and not args.skip_tuning:
        print(f"\n{'-'*80}")
        print(f"STAGE 3: Running models with best hyperparameters")
        print(f"{'-'*80}")

        stage3_start = time.time()

        best_models_results = run_best_models(
            args.data_path,
            tuning_dir,
            best_models_dir,
            train_test_split=args.train_test_split,
            horizons=args.horizons,
            use_gpu=args.use_gpu,
            nnss_epochs=args.nnss_epochs,
            nnss_patience=args.nnss_patience,
            validation_split=args.validation_split,
        )

        stage3_time = time.time() - stage3_start
        print(f"\nStage 3 completed in {stage3_time:.2f} seconds")
    else:
        if args.skip_tuning:
            print(f"\nSkipping Stage 3 (best models) because tuning was skipped")
        else:
            print(f"\nSkipping Stage 3 (best models) as requested")

    print(f"\n{'-'*80}")
    print(f"STAGE 4: Compiling final report")
    print(f"{'-'*80}")

    if args.skip_tuning and args.skip_best_models:
        try:
            from src.utils.report_generator import generate_comprehensive_report

            generate_comprehensive_report(
                results_dir=results_dir,
                baseline_dir=baseline_dir,
                tuning_dir=None,
                best_models_dir=None,
                horizons=args.horizons,
                output_file=results_dir / "baseline_report.html",
            )

            print(f"\nBaseline report saved to {results_dir / 'baseline_report.html'}")
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            print("Continuing with other tasks...")
    else:
        try:
            from src.utils.report_generator import generate_comprehensive_report

            generate_comprehensive_report(
                results_dir=results_dir,
                baseline_dir=baseline_dir,
                tuning_dir=tuning_dir,
                best_models_dir=best_models_dir,
                horizons=args.horizons,
                output_file=results_dir / "comprehensive_report.html",
            )

            print(
                f"\nComprehensive report saved to {results_dir / 'comprehensive_report.html'}"
            )
        except Exception as e:
            print(f"Error generating comprehensive report: {str(e)}")

    if not args.skip_baseline:
        try:
            from src.run_baseline_models import generate_error_boxplots

            print(f"\nGenerating error boxplots...")
            generate_error_boxplots(
                results_dir, horizons=args.horizons, maturities=[1, 5, 7, 10, 30]
            )
            print(f"Error boxplots generated successfully.")
        except Exception as e:
            print(f"Error generating boxplots: {str(e)}")
            print("Continuing with other tasks...")

    total_time = time.time() - start_time
    print(f"\nEvaluation completed in {total_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run yield curve model evaluation")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/ettj_padronizado.csv",
        help="Path to the yield curve data CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save all results",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[5, 20, 60, 120],
        help="Forecast horizons to evaluate",
    )
    parser.add_argument(
        "--train_test_split",
        type=float,
        default=0.8,
        help="Proportion of data to use for training",
    )
    parser.add_argument(
        "--tune_models",
        type=str,
        nargs="+",
        default=["dns", "dnss", "nnss"],
        help="Models to tune hyperparameters for",
    )
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use GPU for NNSS model if available"
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip running baseline models with default configurations",
    )
    parser.add_argument(
        "--skip_tuning",
        action="store_true",
        default=False,
        help="Skip hyperparameter tuning (default: False if best models are run, True otherwise)",
    )
    parser.add_argument(
        "--skip_best_models",
        action="store_true",
        default=False,
        help="Skip running models with best hyperparameters (default: False if tuning is run, True otherwise)",
    )
    parser.add_argument(
        "--nnss_epochs",
        type=int,
        default=500,
        help="Maximum number of epochs for NNSS training (default: 500)",
    )
    parser.add_argument(
        "--nnss_patience",
        type=int,
        default=10,
        help="Patience for early stopping in NNSS training (default: 10)",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.15,
        help="Proportion of training data to use for validation in NNSS (stages 1 & 3) (default: 0.15)",
    )

    args = parser.parse_args()

    if args.skip_best_models and not parser.get_default("skip_tuning"):
        if "--skip_tuning" not in sys.argv:
            args.skip_tuning = True
    elif not args.skip_tuning and parser.get_default("skip_best_models"):
        if "--skip_best_models" not in sys.argv:
            args.skip_best_models = False

    if not 0 < args.validation_split < 1:
        parser.error("--validation_split must be between 0 and 1 (exclusive)")

    main(args)
