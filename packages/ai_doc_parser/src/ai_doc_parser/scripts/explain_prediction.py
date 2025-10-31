"""
Simple function to explain why a specific row was classified the way it was.

This provides a quick way to understand individual predictions without running the full analysis.
"""

import sys
from pathlib import Path

import pandas as pd

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from ai_doc_parser.text_class import CLASS_MAP_INV, TextClass
from ai_doc_parser.training.classifier_trainer import load_model, prepare_df_for_model


def explain_prediction(model_path: str, data_path: str, row_index: int, num_features: int = 10):
    """
    Explain why a specific row was classified the way it was.

    Args:
        model_path: Path to the trained model
        data_path: Path to the data CSV
        row_index: Index of the row to analyze
        num_features: Number of top features to show

    Returns:
        Dictionary with explanation details
    """
    # Load model and data
    model = load_model(model_path)
    df = pd.read_csv(data_path)

    # Prepare data
    df_processed, feature_columns = prepare_df_for_model(df, add_shifted_features=True, verbose=False)
    X = df_processed[feature_columns]
    y_true = df_processed['LabelledClass'] if 'LabelledClass' in df_processed.columns else None

    # Get the specific row
    if row_index >= len(X):
        print(f"Row index {row_index} out of range. Data has {len(X)} rows.")
        return None

    sample = X.iloc[row_index : row_index + 1]
    true_label = y_true.iloc[row_index] if y_true is not None else None

    # Get prediction
    prediction = model.predict(sample)[0]
    prediction_proba = model.predict_proba(sample)[0]

    # Get class names
    pred_class_name = (
        TextClass(CLASS_MAP_INV[prediction]).name if prediction in CLASS_MAP_INV else f"Class_{prediction}"
    )
    true_class_name = (
        TextClass(CLASS_MAP_INV[true_label]).name if true_label in CLASS_MAP_INV else f"Class_{true_label}"
    )

    # Get feature values and importance
    feature_values = sample.iloc[0]
    feature_importance = model.feature_importances_

    # Create feature contribution dataframe
    feature_contrib = pd.DataFrame(
        {'feature': feature_columns, 'value': feature_values.values, 'importance': feature_importance}
    )

    # Sort by importance and get top features
    top_features = feature_contrib.sort_values('importance', ascending=False).head(num_features)

    # Print explanation
    print(f"\n{'='*80}")
    print(f"EXPLANATION FOR ROW {row_index}")
    print(f"{'='*80}")
    print(f"Predicted Class: {pred_class_name} (Class {prediction})")
    print(f"Prediction Confidence: {max(prediction_proba):.3f}")
    if true_label is not None:
        print(f"True Class: {true_class_name} (Class {true_label})")
        print(f"Correct: {'Yes' if prediction == true_label else 'No'}")

    print(f"\nTop {num_features} Contributing Features:")
    print(f"{'Feature':<35} {'Value':<10} {'Importance':<12} {'Contribution':<12}")
    print(f"{'-'*75}")

    for _, row in top_features.iterrows():
        contribution = row['value'] * row['importance']
        print(f"{row['feature']:<35} {row['value']:<10.3f} {row['importance']:<12.4f} {contribution:<12.4f}")

    # Try LIME explanation if available
    try:
        from lime import lime_tabular

        class_names = [TextClass(CLASS_MAP_INV[i]).name for i in sorted(CLASS_MAP_INV.keys())]

        # Create LIME explainer
        lime_explainer = lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=feature_columns,
            class_names=class_names,
            mode='classification',
            discretize_continuous=True,
        )

        # Generate explanation
        explanation = lime_explainer.explain_instance(sample.values[0], model.predict_proba, num_features=num_features)

        print(f"\nLIME Explanation (Top {num_features} features):")
        print(f"{'Feature':<35} {'Weight':<10} {'Effect':<15}")
        print(f"{'-'*65}")

        exp_list = explanation.as_list()
        for feature, weight in exp_list:
            effect = "increases" if weight > 0 else "decreases"
            print(f"{feature:<35} {weight:<10.3f} {effect:<15}")

    except ImportError:
        print("\nLIME not available. Install with: pip install lime for more detailed explanations.")
    except Exception as e:
        print(f"\nLIME explanation failed: {e}")

    # Return explanation data
    return {
        'row_index': row_index,
        'prediction': prediction,
        'pred_class_name': pred_class_name,
        'true_label': true_label,
        'true_class_name': true_class_name,
        'confidence': max(prediction_proba),
        'top_features': top_features,
        'feature_values': feature_values,
        'prediction_proba': prediction_proba,
    }


def compare_predictions(model_path: str, data_path: str, row_indices: list, num_features: int = 10):
    """
    Compare predictions for multiple rows to understand differences.

    Args:
        model_path: Path to the trained model
        data_path: Path to the data CSV
        row_indices: List of row indices to compare
        num_features: Number of top features to show
    """
    print(f"\n{'='*80}")
    print(f"COMPARING PREDICTIONS FOR ROWS: {row_indices}")
    print(f"{'='*80}")

    explanations = []
    for row_idx in row_indices:
        print(f"\n{'-'*40}")
        print(f"ROW {row_idx}")
        print(f"{'-'*40}")
        explanation = explain_prediction(model_path, data_path, row_idx, num_features)
        if explanation:
            explanations.append(explanation)

    # Compare top features across rows
    if len(explanations) > 1:
        print(f"\n{'='*80}")
        print("FEATURE COMPARISON ACROSS ROWS")
        print(f"{'='*80}")

        # Get all unique features from top features
        all_features = set()
        for exp in explanations:
            all_features.update(exp['top_features']['feature'].tolist())

        # Create comparison table
        print(f"{'Feature':<35}", end="")
        for exp in explanations:
            print(f"{'Row ' + str(exp['row_index']):<15}", end="")
        print()
        print(f"{'-'*80}")

        for feature in sorted(all_features):
            print(f"{feature:<35}", end="")
            for exp in explanations:
                feature_data = exp['top_features'][exp['top_features']['feature'] == feature]
                if not feature_data.empty:
                    value = feature_data['value'].iloc[0]
                    print(f"{value:<15.3f}", end="")
                else:
                    print(f"{'N/A':<15}", end="")
            print()

    return explanations


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explain why a specific row was classified the way it was")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("--row_index", type=int, help="Index of the row to analyze")
    parser.add_argument("--row_indices", type=str, help="Comma-separated list of row indices to compare")
    parser.add_argument("--num_features", type=int, default=10, help="Number of top features to show")

    args = parser.parse_args()

    if args.row_index is not None:
        explain_prediction(args.model_path, args.data_path, args.row_index, args.num_features)
    elif args.row_indices is not None:
        row_indices = [int(x.strip()) for x in args.row_indices.split(',')]
        compare_predictions(args.model_path, args.data_path, row_indices, args.num_features)
    else:
        print("Please specify either --row_index or --row_indices")
