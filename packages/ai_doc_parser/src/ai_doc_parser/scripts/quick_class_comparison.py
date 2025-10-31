"""
Quick function to compare probabilities between two specific classes.

This provides a simple way to understand what drives the probability difference
between any two classes without running the full analysis.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from ai_doc_parser.text_class import CLASS_MAP_INV, TextClass
from ai_doc_parser.training.classifier_trainer import load_model, prepare_df_for_model


def compare_two_classes(
    model_path: str, data_path: str, row_index: int, class1_name: str, class2_name: str, num_features: int = 10
):
    """
    Quick comparison of probabilities between two specific classes.

    Args:
        model_path: Path to the trained model
        data_path: Path to the data CSV
        row_index: Index of the row to analyze
        class1_name: First class name (e.g., "HEADING")
        class2_name: Second class name (e.g., "HEADING_CONT")
        num_features: Number of top features to show

    Returns:
        Dictionary with comparison results
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

    # Get prediction and probabilities
    prediction = model.predict(sample)[0]
    prediction_proba = model.predict_proba(sample)[0]

    # Get class names and indices
    class_names = [TextClass(CLASS_MAP_INV[i]).name for i in sorted(CLASS_MAP_INV.keys())]
    pred_class_name = (
        TextClass(CLASS_MAP_INV[prediction]).name if prediction in CLASS_MAP_INV else f"Class_{prediction}"
    )
    true_class_name = (
        TextClass(CLASS_MAP_INV[true_label]).name if true_label in CLASS_MAP_INV else f"Class_{true_label}"
    )

    # Find class indices
    class1_idx = None
    class2_idx = None

    for i, class_name in enumerate(class_names):
        if class_name == class1_name:
            class1_idx = i
        elif class_name == class2_name:
            class2_idx = i

    if class1_idx is None or class2_idx is None:
        print(f"Could not find classes {class1_name} and {class2_name}")
        print(f"Available classes: {class_names}")
        return None

    # Get probabilities for the two classes
    class1_prob = prediction_proba[class1_idx]
    class2_prob = prediction_proba[class2_idx]
    prob_difference = class2_prob - class1_prob

    print(f"\n{'='*70}")
    print(f"CLASS PROBABILITY COMPARISON: {class1_name} vs {class2_name}")
    print(f"{'='*70}")
    print(f"Row {row_index}: {pred_class_name} (True: {true_class_name})")
    print(f"{class1_name} probability: {class1_prob:.4f}")
    print(f"{class2_name} probability: {class2_prob:.4f}")
    print(f"Difference ({class2_name} - {class1_name}): {prob_difference:.4f}")

    if prob_difference > 0:
        print(f"→ {class2_name} is more likely by {prob_difference:.4f}")
    else:
        print(f"→ {class1_name} is more likely by {abs(prob_difference):.4f}")

    # SHAP analysis for the two classes
    try:
        import shap

        # Create SHAP explainer
        shap_explainer = shap.TreeExplainer(model)
        shap_values = shap_explainer.shap_values(sample)

        # Handle multi-class case
        if isinstance(shap_values, list):
            class1_shap = shap_values[class1_idx]
            class2_shap = shap_values[class2_idx]
        else:
            class1_shap = shap_values[0]
            class2_shap = shap_values[0]

        # Calculate SHAP difference
        shap_difference = class2_shap - class1_shap

        # Create feature comparison dataframe
        feature_comparison = pd.DataFrame(
            {
                'feature': feature_columns,
                'value': sample.values[0],
                f'{class1_name}_shap': class1_shap,
                f'{class2_name}_shap': class2_shap,
                'shap_difference': shap_difference,
            }
        )

        # Sort by absolute SHAP difference
        feature_comparison = feature_comparison.sort_values('shap_difference', key=abs, ascending=False)

        print(f"\nTop {num_features} Features Driving Class Difference:")
        print(f"{'Feature':<30} {'Value':<8} {'Difference':<12} {'Effect':<20}")
        print(f"{'-'*75}")

        for _, row in feature_comparison.head(num_features).iterrows():
            effect = f"Favors {class2_name}" if row['shap_difference'] > 0 else f"Favors {class1_name}"
            print(f"{row['feature']:<30} {row['value']:<8.3f} {row['shap_difference']:<12.3f} {effect:<20}")

        return {
            'row_index': row_index,
            'class1_name': class1_name,
            'class2_name': class2_name,
            'class1_prob': class1_prob,
            'class2_prob': class2_prob,
            'prob_difference': prob_difference,
            'predicted_class': pred_class_name,
            'true_class': true_class_name,
            'feature_comparison': feature_comparison,
        }

    except ImportError:
        print("\nSHAP not available. Install with: pip install shap for detailed analysis.")
        return {
            'row_index': row_index,
            'class1_name': class1_name,
            'class2_name': class2_name,
            'class1_prob': class1_prob,
            'class2_prob': class2_prob,
            'prob_difference': prob_difference,
            'predicted_class': pred_class_name,
            'true_class': true_class_name,
        }
    except Exception as e:
        print(f"\nSHAP analysis failed: {e}")
        return {
            'row_index': row_index,
            'class1_name': class1_name,
            'class2_name': class2_name,
            'class1_prob': class1_prob,
            'class2_prob': class2_prob,
            'prob_difference': prob_difference,
            'predicted_class': pred_class_name,
            'true_class': true_class_name,
        }


def analyze_class_decision_boundary(
    model_path: str, data_path: str, class1_name: str, class2_name: str, num_samples: int = 20
):
    """
    Analyze the decision boundary between two classes by looking at multiple samples.
    """
    # Load model and data
    model = load_model(model_path)
    df = pd.read_csv(data_path)

    # Prepare data
    df_processed, feature_columns = prepare_df_for_model(df, add_shifted_features=True, verbose=False)
    X = df_processed[feature_columns]
    y_true = df_processed['LabelledClass'] if 'LabelledClass' in df_processed.columns else None

    # Get class names and indices
    class_names = [TextClass(CLASS_MAP_INV[i]).name for i in sorted(CLASS_MAP_INV.keys())]

    class1_idx = None
    class2_idx = None

    for i, class_name in enumerate(class_names):
        if class_name == class1_name:
            class1_idx = i
        elif class_name == class2_name:
            class2_idx = i

    if class1_idx is None or class2_idx is None:
        print(f"Could not find classes {class1_name} and {class2_name}")
        return None

    # Sample random rows
    sample_indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)

    print(f"\n{'='*70}")
    print(f"DECISION BOUNDARY ANALYSIS: {class1_name} vs {class2_name}")
    print(f"{'='*70}")

    results = []
    for row_idx in sample_indices:
        sample = X.iloc[row_idx : row_idx + 1]
        prediction = model.predict(sample)[0]
        prediction_proba = model.predict_proba(sample)[0]

        class1_prob = prediction_proba[class1_idx]
        class2_prob = prediction_proba[class2_idx]
        prob_difference = class2_prob - class1_prob

        pred_class_name = (
            TextClass(CLASS_MAP_INV[prediction]).name if prediction in CLASS_MAP_INV else f"Class_{prediction}"
        )
        true_class_name = TextClass(CLASS_MAP_INV[y_true.iloc[row_idx]]).name if y_true is not None else "Unknown"

        results.append(
            {
                'row_index': row_idx,
                'predicted': pred_class_name,
                'true': true_class_name,
                'class1_prob': class1_prob,
                'class2_prob': class2_prob,
                'prob_difference': prob_difference,
            }
        )

    # Sort by probability difference
    results.sort(key=lambda x: x['prob_difference'])

    print(f"{'Row':<5} {'Predicted':<15} {'True':<15} {f'{class1_name}':<12} {f'{class2_name}':<12} {'Difference':<12}")
    print(f"{'-'*80}")

    for result in results:
        print(
            f"{result['row_index']:<5} {result['predicted']:<15} {result['true']:<15} {result['class1_prob']:<12.4f} {result['class2_prob']:<12.4f} {result['prob_difference']:<12.4f}"
        )

    # Find the decision boundary (where probabilities are closest)
    closest_to_boundary = min(results, key=lambda x: abs(x['prob_difference']))

    print(f"\nClosest to decision boundary: Row {closest_to_boundary['row_index']}")
    print(f"  {class1_name}: {closest_to_boundary['class1_prob']:.4f}")
    print(f"  {class2_name}: {closest_to_boundary['class2_prob']:.4f}")
    print(f"  Difference: {closest_to_boundary['prob_difference']:.4f}")

    return results


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick comparison of probabilities between two classes")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("--row_index", type=int, help="Index of the row to analyze")
    parser.add_argument("--class1", type=str, required=True, help="First class name (e.g., HEADING)")
    parser.add_argument("--class2", type=str, required=True, help="Second class name (e.g., HEADING_CONT)")
    parser.add_argument("--num_features", type=int, default=10, help="Number of top features to show")
    parser.add_argument(
        "--boundary_analysis", action="store_true", help="Analyze decision boundary across multiple samples"
    )

    args = parser.parse_args()

    if args.boundary_analysis:
        analyze_class_decision_boundary(args.model_path, args.data_path, args.class1, args.class2)
    elif args.row_index is not None:
        compare_two_classes(
            args.model_path, args.data_path, args.row_index, args.class1, args.class2, args.num_features
        )
    else:
        print("Please specify either --row_index or --boundary_analysis")
