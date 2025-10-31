"""
Quick example showing how to use the Model Interpretability Analyzer
with your existing RandomForestClassifier training workflow.

This example demonstrates the basic usage without command-line arguments.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from ai_doc_parser.text_class import CLASS_MAP_INV, TextClass
from ai_doc_parser.tools.model_interpretability import ModelInterpretabilityAnalyzer
from ai_doc_parser.training.classifier_trainer import load_model, train_multiclass

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def quick_analysis_example():
    """
    Example showing how to analyze a trained model's predictions.
    """
    log.info("Starting quick analysis example...")

    # Example 1: Analyze a model you just trained
    log.info("=" * 60)
    log.info("EXAMPLE 1: Analyzing a freshly trained model")
    log.info("=" * 60)

    # Load your training data (replace with your actual data path)
    # data_path = "path/to/your/training_data.csv"
    # df = pd.read_csv(data_path)

    # For this example, we'll create some dummy data
    # In practice, you would load your actual training data
    log.info("Creating dummy data for demonstration...")
    n_samples = 1000
    n_features = 20

    # Create dummy features (replace with your actual feature columns)
    feature_columns = [
        "is_bold",
        "is_italic",
        "last_char_isdigit",
        "first_char_bullet",
        "first_word_compound",
        "starts_with_keyword",
        "number_dots",
        "is_title_case",
        "all_caps",
        "ends_with_period",
        "ends_with_question_mark",
        "ends_with_exclamation",
        "ends_with_dash_char",
        "ends_with_punctuation",
        "ends_with_letter",
        "ends_with_special_char",
        "ends_with_digit",
        "last_word_is_roman",
        "ends_with_upper",
        "num_consecutive_spaces",
    ]

    # Create dummy data
    np.random.seed(42)
    X_dummy = pd.DataFrame(
        np.random.rand(n_samples, n_features), columns=feature_columns
    )

    # Create dummy labels (replace with your actual labels)
    y_dummy = pd.Series(np.random.randint(0, 5, n_samples))  # 5 classes

    # Train a model (replace with your actual training)
    log.info("Training a dummy model...")
    x_train, x_test, y_train, y_test, model, feature_columns = train_multiclass(
        pd.concat([X_dummy, y_dummy.rename("LabelledClass")], axis=1),
        test_size=0.2,
        random_state=42,
    )

    # Initialize the analyzer
    class_names = [
        TextClass(CLASS_MAP_INV[i]).name for i in sorted(CLASS_MAP_INV.keys())
    ]
    analyzer = ModelInterpretabilityAnalyzer(model, feature_columns, class_names)

    # Run feature importance analysis
    log.info("Running feature importance analysis...")
    importance_results = analyzer.analyze_feature_importance(
        top_n=10, save_plot=True, output_dir="quick_analysis_output"
    )

    # Show top features
    log.info("Top 10 most important features:")
    for _, row in importance_results["top_features"].iterrows():
        log.info(f"  {row['feature']:<25}: {row['importance']:.4f}")

    # Example 2: Analyze specific predictions
    log.info("\n" + "=" * 60)
    log.info("EXAMPLE 2: Analyzing specific predictions")
    log.info("=" * 60)

    # Analyze first 3 test samples
    test_samples = x_test.head(3)
    test_labels = y_test.head(3)

    for i, (idx, sample) in enumerate(test_samples.iterrows()):
        log.info(f"\nAnalyzing sample {i+1}:")

        # Get prediction
        prediction = model.predict([sample.values])[0]
        prediction_proba = model.predict_proba([sample.values])[0]
        true_label = test_labels.iloc[i]

        pred_class_name = (
            TextClass(CLASS_MAP_INV[prediction]).name
            if prediction in CLASS_MAP_INV
            else f"Class_{prediction}"
        )
        true_class_name = (
            TextClass(CLASS_MAP_INV[true_label]).name
            if true_label in CLASS_MAP_INV
            else f"Class_{true_label}"
        )

        log.info(f"  True Class: {true_class_name}")
        log.info(f"  Predicted Class: {pred_class_name}")
        log.info(f"  Confidence: {max(prediction_proba):.3f}")
        log.info(f"  Correct: {'Yes' if prediction == true_label else 'No'}")

        # Show top contributing features for this sample
        feature_importance = model.feature_importances_
        feature_contrib = pd.DataFrame(
            {
                "feature": feature_columns,
                "value": sample.values,
                "importance": feature_importance,
            }
        ).sort_values("importance", ascending=False)

        log.info(f"  Top 5 contributing features:")
        for _, row in feature_contrib.head(5).iterrows():
            log.info(
                f"    {row['feature']:<20}: value={row['value']:.3f}, importance={row['importance']:.4f}"
            )

    # Example 3: Load and analyze an existing model
    log.info("\n" + "=" * 60)
    log.info("EXAMPLE 3: Loading and analyzing an existing model")
    log.info("=" * 60)

    # Save the model first (for demonstration)
    model_path = "quick_analysis_output/dummy_model.sav"
    Path("quick_analysis_output").mkdir(exist_ok=True)
    import joblib

    joblib.dump(model, model_path)

    # Load the model
    log.info(f"Loading model from: {model_path}")
    loaded_model = load_model(model_path)

    # Initialize analyzer with loaded model
    analyzer_loaded = ModelInterpretabilityAnalyzer(
        loaded_model, feature_columns, class_names
    )

    # Run a quick analysis
    log.info("Running quick analysis on loaded model...")
    quick_results = analyzer_loaded.analyze_feature_importance(
        top_n=5, save_plot=True, output_dir="quick_analysis_output"
    )

    log.info("Analysis completed successfully!")
    log.info("Check the 'quick_analysis_output' directory for saved plots and results.")


def analyze_existing_model_example():
    """
    Example showing how to analyze an existing trained model.
    """
    log.info("=" * 60)
    log.info("EXAMPLE: Analyzing an existing model")
    log.info("=" * 60)

    # Replace these paths with your actual model and data paths
    model_path = "data/models/RandomForestClassifier.sav"  # Your model path
    data_path = "shifted_labelled_pdf.csv"  # Your data path

    # Check if files exist
    if not Path(model_path).exists():
        log.warning(f"Model file not found: {model_path}")
        log.info("Please update the model_path variable with your actual model path")
        return

    if not Path(data_path).exists():
        log.warning(f"Data file not found: {data_path}")
        log.info("Please update the data_path variable with your actual data path")
        return

    try:
        # Load the model
        log.info(f"Loading model from: {model_path}")
        model = load_model(model_path)

        # Load and prepare data
        log.info(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)

        # Prepare data (you might need to adjust this based on your data structure)
        from ai_doc_parser.training.classifier_trainer import prepare_df_for_model

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

        # Initialize analyzer
        class_names = [
            TextClass(CLASS_MAP_INV[i]).name for i in sorted(CLASS_MAP_INV.keys())
        ]
        analyzer = ModelInterpretabilityAnalyzer(model, feature_columns, class_names)

        # Run comprehensive analysis
        log.info("Running comprehensive analysis...")
        results = analyzer.comprehensive_analysis(
            X=X, y_true=y_true, output_dir="existing_model_analysis"
        )

        log.info("Analysis completed successfully!")
        log.info("Check the 'existing_model_analysis' directory for all results.")

    except Exception as e:
        log.error(f"Error during analysis: {e}")
        log.info(
            "Make sure your model and data paths are correct and the data format matches expectations."
        )


if __name__ == "__main__":
    import numpy as np

    # Run the quick analysis example
    quick_analysis_example()

    # Uncomment the line below to analyze an existing model
    # analyze_existing_model_example()
