"""
Example script demonstrating how to use the Model Interpretability Analyzer
to understand why certain rows are being classified the way they are.

This script shows how to:
1. Load a trained model and data
2. Run comprehensive interpretability analysis
3. Analyze specific predictions
4. Generate reports and visualizations

Usage:
    python analyze_model_predictions.py --model_path path/to/model.sav --data_path path/to/data.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from ai_doc_parser.text_class import CLASS_MAP_INV, TextClass
from ai_doc_parser.tools.model_interpretability import (
    ModelInterpretabilityAnalyzer,
)
from ai_doc_parser.training.classifier_trainer import load_model, prepare_df_for_model

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def analyze_specific_predictions(
    analyzer: ModelInterpretabilityAnalyzer,
    X: pd.DataFrame,
    y_true: Optional[pd.Series] = None,
    sample_indices: list = None,
    output_dir: str = "analysis_output",
):
    """
    Analyze specific predictions in detail.

    Args:
        analyzer: Initialized ModelInterpretabilityAnalyzer
        X: Feature matrix
        y_true: True labels (optional)
        sample_indices: Specific samples to analyze (default: first 5)
        output_dir: Directory to save outputs
    """
    if sample_indices is None:
        sample_indices = list(range(min(5, len(X))))

    log.info(f"Analyzing specific predictions for samples: {sample_indices}")

    results = {}

    for idx in sample_indices:
        if idx >= len(X):
            continue

        log.info(f"\n{'='*60}")
        log.info(f"ANALYZING SAMPLE {idx}")
        log.info(f"{'='*60}")

        sample = X.iloc[idx : idx + 1]  # Keep as DataFrame for consistency

        # Get prediction
        prediction = analyzer.model.predict(sample)[0]
        prediction_proba = analyzer.model.predict_proba(sample)[0]

        # Get class name
        pred_class_name = (
            TextClass(CLASS_MAP_INV[prediction]).name
            if prediction in CLASS_MAP_INV
            else f"Class_{prediction}"
        )

        log.info(f"Predicted Class: {pred_class_name} (Class {prediction})")
        log.info(f"Prediction Confidence: {max(prediction_proba):.3f}")

        if y_true is not None:
            true_label = y_true.iloc[idx]
            true_class_name = (
                TextClass(CLASS_MAP_INV[true_label]).name
                if true_label in CLASS_MAP_INV
                else f"Class_{true_label}"
            )
            log.info(f"True Class: {true_class_name} (Class {true_label})")
            log.info(f"Correct: {'Yes' if prediction == true_label else 'No'}")

        # Show top contributing features
        feature_values = sample.iloc[0]
        feature_importance = analyzer.model.feature_importances_

        # Create feature contribution dataframe
        feature_contrib = pd.DataFrame(
            {
                "feature": analyzer.feature_names,
                "value": feature_values.values,
                "importance": feature_importance,
            }
        )

        # Sort by importance and show top 10
        top_features = feature_contrib.sort_values("importance", ascending=False).head(
            10
        )

        log.info(f"\nTop 10 Contributing Features:")
        log.info(f"{'Feature':<25} {'Value':<10} {'Importance':<12}")
        log.info(f"{'-'*50}")
        for _, row in top_features.iterrows():
            log.info(
                f"{row['feature']:<25} {row['value']:<10.3f} {row['importance']:<12.4f}"
            )

        # LIME explanation if available
        if hasattr(analyzer, "lime_explainer") and analyzer.lime_explainer is not None:
            try:
                explanation = analyzer.lime_explainer.explain_instance(
                    sample.values[0], analyzer.model.predict_proba, num_features=10
                )

                log.info(f"\nLIME Explanation:")
                exp_list = explanation.as_list()
                for feature, weight in exp_list:
                    direction = "increases" if weight > 0 else "decreases"
                    log.info(f"  {feature}: {weight:.3f} ({direction} probability)")

            except Exception as e:
                log.warning(f"Could not generate LIME explanation: {e}")

        results[idx] = {
            "prediction": prediction,
            "prediction_proba": prediction_proba,
            "pred_class_name": pred_class_name,
            "true_label": true_label if y_true is not None else None,
            "true_class_name": true_class_name if y_true is not None else None,
            "top_features": top_features,
        }

    return results


def generate_analysis_report(
    analyzer: ModelInterpretabilityAnalyzer,
    analysis_results: dict,
    output_dir: str = "analysis_output",
):
    """
    Generate a comprehensive analysis report.

    Args:
        analyzer: Initialized ModelInterpretabilityAnalyzer
        analysis_results: Results from comprehensive analysis
        output_dir: Directory to save the report
    """
    log.info("Generating analysis report...")

    report_path = Path(output_dir) / "analysis_report.txt"

    with open(report_path, "w") as f:
        f.write("MODEL INTERPRETABILITY ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")

        # Model information
        f.write("MODEL INFORMATION:\n")
        f.write(f"Model Type: {type(analyzer.model).__name__}\n")
        f.write(f"Number of Features: {len(analyzer.feature_names)}\n")
        f.write(f"Number of Classes: {len(analyzer.class_names)}\n")
        f.write(f"Classes: {analyzer.class_names}\n\n")

        # Feature importance summary
        if "feature_importance" in analysis_results:
            f.write("FEATURE IMPORTANCE SUMMARY:\n")
            f.write("-" * 30 + "\n")
            top_features = analysis_results["feature_importance"]["top_features"]
            for _, row in top_features.head(10).iterrows():
                f.write(f"{row['feature']:<25}: {row['importance']:.4f}\n")
            f.write("\n")

        # SHAP analysis summary
        if "shap" in analysis_results and analysis_results["shap"]:
            f.write("SHAP ANALYSIS SUMMARY:\n")
            f.write("-" * 25 + "\n")
            shap_importance = analysis_results["shap"]["feature_importance"]
            for _, row in shap_importance.head(10).iterrows():
                f.write(f"{row['feature']:<25}: {row['shap_importance']:.4f}\n")
            f.write("\n")

        # Error analysis summary
        if "error_analysis" in analysis_results:
            f.write("ERROR ANALYSIS SUMMARY:\n")
            f.write("-" * 25 + "\n")
            error_analysis = analysis_results["error_analysis"]
            f.write(f"Total Samples: {error_analysis['total_samples']}\n")
            f.write(f"Total Errors: {error_analysis['total_errors']}\n")
            f.write(f"Error Rate: {error_analysis['error_rate']:.3f}\n")
            f.write(
                f"Mean Confidence (Errors): {error_analysis['error_confidences'].mean():.3f}\n"
            )
            f.write(
                f"Mean Confidence (Correct): {error_analysis['correct_confidences'].mean():.3f}\n\n"
            )

        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 15 + "\n")
        f.write(
            "1. Focus on the top contributing features identified in the analysis\n"
        )
        f.write("2. Investigate samples with low prediction confidence\n")
        f.write("3. Consider feature engineering based on SHAP and LIME insights\n")
        f.write("4. Review misclassified samples to understand model limitations\n")

    log.info(f"Analysis report saved to: {report_path}")


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze model predictions using interpretability tools"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model file"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the CSV data file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--sample_size", type=int, default=100, help="Sample size for SHAP analysis"
    )
    parser.add_argument(
        "--analyze_samples",
        type=str,
        help="Comma-separated list of sample indices to analyze in detail",
    )
    parser.add_argument(
        "--skip_shap", action="store_true", help="Skip SHAP analysis (faster)"
    )
    parser.add_argument(
        "--skip_lime", action="store_true", help="Skip LIME analysis (faster)"
    )

    args = parser.parse_args()

    # Load data
    log.info(f"Loading data from: {args.data_path}")
    df = pd.read_csv(args.data_path)

    # Prepare data for analysis
    log.info("Preparing data for analysis...")
    df_processed, feature_columns = prepare_df_for_model(
        df, add_shifted_features=True, verbose=True
    )

    # Separate features and labels
    X = df_processed[feature_columns]
    y_true = (
        df_processed["LabelledClass"]
        if "LabelledClass" in df_processed.columns
        else None
    )

    log.info(f"Data shape: {X.shape}")
    log.info(f"Number of features: {len(feature_columns)}")

    # Load model
    log.info(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path)

    # Initialize analyzer
    class_names = [
        TextClass(CLASS_MAP_INV[i]).name for i in sorted(CLASS_MAP_INV.keys())
    ]
    analyzer = ModelInterpretabilityAnalyzer(model, feature_columns, class_names)

    # Run comprehensive analysis
    log.info("Starting comprehensive analysis...")
    analysis_results = {}

    # Feature importance analysis
    log.info("Running feature importance analysis...")
    analysis_results["feature_importance"] = analyzer.analyze_feature_importance(
        top_n=20, save_plot=True, output_dir=args.output_dir
    )

    # Tree structure analysis
    log.info("Running tree structure analysis...")
    analysis_results["tree_structure"] = analyzer.analyze_tree_structure(
        tree_index=0, max_depth=3, save_text=True, output_dir=args.output_dir
    )

    # SHAP analysis (if not skipped)
    if not args.skip_shap:
        log.info("Running SHAP analysis...")
        analysis_results["shap"] = analyzer.analyze_shap_values(
            X=X,
            sample_size=args.sample_size,
            save_plots=True,
            output_dir=args.output_dir,
        )
    else:
        log.info("Skipping SHAP analysis as requested")

    # LIME analysis (if not skipped)
    if not args.skip_lime:
        log.info("Running LIME analysis...")
        analysis_results["lime"] = analyzer.analyze_lime_explanations(
            X=X, save_plots=True, output_dir=args.output_dir
        )
    else:
        log.info("Skipping LIME analysis as requested")

    # Error analysis (if true labels available)
    if y_true is not None:
        log.info("Running error analysis...")
        analysis_results["error_analysis"] = analyzer.analyze_prediction_errors(
            X=X, y_true=y_true, save_plots=True, output_dir=args.output_dir
        )
    else:
        log.info("Skipping error analysis - no true labels available")

    # Analyze specific samples if requested
    if args.analyze_samples:
        sample_indices = [int(x.strip()) for x in args.analyze_samples.split(",")]
        log.info(f"Analyzing specific samples: {sample_indices}")
        specific_results = analyze_specific_predictions(
            analyzer, X, y_true, sample_indices, args.output_dir
        )
        analysis_results["specific_predictions"] = specific_results

    # Generate comprehensive report
    generate_analysis_report(analyzer, analysis_results, args.output_dir)

    log.info("Analysis completed successfully!")
    log.info(f"All results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
