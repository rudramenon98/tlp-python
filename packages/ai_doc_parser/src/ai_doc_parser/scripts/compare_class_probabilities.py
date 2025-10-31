"""
Script to analyze probability differences between two specific classes.

This tool helps you understand what features drive the probability difference
between any two classes (e.g., HEADER vs HEADER_CONT, PARAGRAPH vs TOC, etc.).
"""

import sys
from pathlib import Path

import pandas as pd

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from ai_doc_parser.text_class import CLASS_MAP_INV, TextClass
from ai_doc_parser.training.classifier_trainer import load_model, prepare_df_for_model


def compare_class_probabilities(
    model_path: str, data_path: str, row_index: int, class1_name: str, class2_name: str, num_features: int = 15
):
    """
    Compare probability differences between two specific classes.

    Args:
        model_path: Path to the trained model
        data_path: Path to the data CSV
        row_index: Index of the row to analyze
        class1_name: First class to compare (e.g., "HEADING")
        class2_name: Second class to compare (e.g., "HEADING_CONT")
        num_features: Number of top features to show
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
        return

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
        return

    # Get probabilities for the two classes
    class1_prob = prediction_proba[class1_idx]
    class2_prob = prediction_proba[class2_idx]
    prob_difference = class2_prob - class1_prob

    print(f"\n{'='*80}")
    print(f"CLASS PROBABILITY COMPARISON FOR ROW {row_index}")
    print(f"{'='*80}")
    print(f"Predicted Class: {pred_class_name} (Class {prediction})")
    print(f"True Class: {true_class_name} (Class {true_label})")
    print(f"Prediction Confidence: {max(prediction_proba):.3f}")

    print(f"\n{'-'*50}")
    print(f"PROBABILITY COMPARISON: {class1_name} vs {class2_name}")
    print(f"{'-'*50}")
    print(f"{class1_name} probability: {class1_prob:.4f}")
    print(f"{class2_name} probability: {class2_prob:.4f}")
    print(f"Difference ({class2_name} - {class1_name}): {prob_difference:.4f}")

    if prob_difference > 0:
        print(f"→ {class2_name} is more likely by {prob_difference:.4f}")
    else:
        print(f"→ {class1_name} is more likely by {abs(prob_difference):.4f}")

    # SHAP analysis for the two classes
    print(f"\n{'='*80}")
    print("SHAP ANALYSIS FOR CLASS DIFFERENCE")
    print(f"{'='*80}")

    try:
        import shap

        # Create SHAP explainer
        shap_explainer = shap.TreeExplainer(model)
        shap_values = shap_explainer.shap_values(sample)

        # Handle multi-class case
        if isinstance(shap_values, list):
            class1_shap = shap_values[class1_idx]
            class2_shap = shap_values[class2_idx]
            class1_expected = shap_explainer.expected_value[class1_idx]
            class2_expected = shap_explainer.expected_value[class2_idx]
        else:
            class1_shap = shap_values[0]
            class2_shap = shap_values[0]
            class1_expected = shap_explainer.expected_value
            class2_expected = shap_explainer.expected_value

        # Calculate SHAP difference
        shap_difference = class2_shap - class1_shap

        print(f"SHAP Expected Values:")
        print(f"  {class1_name}: {class1_expected:.4f}")
        print(f"  {class2_name}: {class2_expected:.4f}")
        print(f"  Difference: {class2_expected - class1_expected:.4f}")

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
        print(
            f"{'Feature':<35} {'Value':<8} {f'{class1_name}':<12} {f'{class2_name}':<12} {'Difference':<12} {'Effect':<15}"
        )
        print(f"{'-'*100}")

        for _, row in feature_comparison.head(num_features).iterrows():
            effect = f"Favors {class2_name}" if row['shap_difference'] > 0 else f"Favors {class1_name}"
            print(
                f"{row['feature']:<35} {row['value']:<8.3f} {row[f'{class1_name}_shap']:<12.3f} {row[f'{class2_name}_shap']:<12.3f} {row['shap_difference']:<12.3f} {effect:<15}"
            )

        # Create SHAP comparison plot
        Path("class_comparison_output").mkdir(exist_ok=True)

        # Waterfall plot for class1
        shap.waterfall_plot(class1_expected, class1_shap, sample.iloc[0], show=False, max_display=num_features)
        import matplotlib.pyplot as plt

        plt.title(f'SHAP Waterfall: {class1_name}')
        plt.savefig(
            f"class_comparison_output/{class1_name.lower()}_waterfall_row_{row_index}.png", dpi=300, bbox_inches='tight'
        )
        plt.close()

        # Waterfall plot for class2
        shap.waterfall_plot(class2_expected, class2_shap, sample.iloc[0], show=False, max_display=num_features)
        plt.title(f'SHAP Waterfall: {class2_name}')
        plt.savefig(
            f"class_comparison_output/{class2_name.lower()}_waterfall_row_{row_index}.png", dpi=300, bbox_inches='tight'
        )
        plt.close()

        print(f"\nSHAP waterfall plots saved to class_comparison_output/")

    except ImportError:
        print("SHAP not available. Install with: pip install shap")
    except Exception as e:
        print(f"SHAP analysis failed: {e}")

    # LIME analysis for class difference
    print(f"\n{'='*80}")
    print("LIME ANALYSIS FOR CLASS DIFFERENCE")
    print(f"{'='*80}")

    try:
        from lime import lime_tabular

        # Create LIME explainer
        lime_explainer = lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=feature_columns,
            class_names=class_names,
            mode='classification',
            discretize_continuous=True,
        )

        # Generate explanations for both classes
        explanation1 = lime_explainer.explain_instance(
            sample.values[0], model.predict_proba, num_features=num_features, labels=[class1_idx]
        )

        explanation2 = lime_explainer.explain_instance(
            sample.values[0], model.predict_proba, num_features=num_features, labels=[class2_idx]
        )

        # Get explanation lists
        exp1_list = explanation1.as_list(label=class1_idx)
        exp2_list = explanation2.as_list(label=class2_idx)

        # Create comparison dataframe
        lime_comparison = []
        all_features = set([item[0] for item in exp1_list] + [item[0] for item in exp2_list])

        for feature in all_features:
            weight1 = next((item[1] for item in exp1_list if item[0] == feature), 0)
            weight2 = next((item[1] for item in exp2_list if item[0] == feature), 0)
            weight_diff = weight2 - weight1

            lime_comparison.append(
                {
                    'feature': feature,
                    f'{class1_name}_weight': weight1,
                    f'{class2_name}_weight': weight2,
                    'weight_difference': weight_diff,
                }
            )

        lime_df = pd.DataFrame(lime_comparison)
        lime_df = lime_df.sort_values('weight_difference', key=abs, ascending=False)

        print(f"LIME Weight Comparison (Top {num_features}):")
        print(f"{'Feature':<35} {f'{class1_name}':<12} {f'{class2_name}':<12} {'Difference':<12} {'Effect':<15}")
        print(f"{'-'*90}")

        for _, row in lime_df.head(num_features).iterrows():
            effect = f"Favors {class2_name}" if row['weight_difference'] > 0 else f"Favors {class1_name}"
            print(
                f"{row['feature']:<35} {row[f'{class1_name}_weight']:<12.3f} {row[f'{class2_name}_weight']:<12.3f} {row['weight_difference']:<12.3f} {effect:<15}"
            )

    except ImportError:
        print("LIME not available. Install with: pip install lime")
    except Exception as e:
        print(f"LIME analysis failed: {e}")

    # Feature importance analysis
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*80}")

    feature_values = sample.iloc[0]
    feature_importance = model.feature_importances_

    # Create feature analysis dataframe
    feature_analysis = pd.DataFrame(
        {'feature': feature_columns, 'value': feature_values.values, 'importance': feature_importance}
    )

    # Sort by importance
    feature_analysis = feature_analysis.sort_values('importance', ascending=False)

    print(f"Top {num_features} Most Important Features:")
    print(f"{'Feature':<35} {'Value':<8} {'Importance':<12} {'Contribution':<12}")
    print(f"{'-'*70}")

    for _, row in feature_analysis.head(num_features).iterrows():
        contribution = row['value'] * row['importance']
        print(f"{row['feature']:<35} {row['value']:<8.3f} {row['importance']:<12.4f} {contribution:<12.4f}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print(f"Row {row_index} Analysis:")
    print(f"  Predicted: {pred_class_name}")
    print(f"  True: {true_class_name}")
    print(f"  {class1_name} probability: {class1_prob:.4f}")
    print(f"  {class2_name} probability: {class2_prob:.4f}")
    print(f"  Difference: {prob_difference:.4f}")

    if prob_difference > 0.1:
        print(f"  → Strong preference for {class2_name}")
    elif prob_difference > 0.05:
        print(f"  → Moderate preference for {class2_name}")
    elif prob_difference > 0.01:
        print(f"  → Slight preference for {class2_name}")
    elif prob_difference > -0.01:
        print(f"  → Very close probabilities")
    elif prob_difference > -0.05:
        print(f"  → Slight preference for {class1_name}")
    elif prob_difference > -0.1:
        print(f"  → Moderate preference for {class1_name}")
    else:
        print(f"  → Strong preference for {class1_name}")

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


def compare_multiple_rows(model_path: str, data_path: str, row_indices: list, class1_name: str, class2_name: str):
    """
    Compare class probabilities across multiple rows.
    """
    print(f"\n{'='*80}")
    print(f"COMPARING {class1_name} vs {class2_name} ACROSS MULTIPLE ROWS")
    print(f"{'='*80}")

    results = []
    for row_idx in row_indices:
        print(f"\n{'-'*40}")
        print(f"ROW {row_idx}")
        print(f"{'-'*40}")

        result = compare_class_probabilities(model_path, data_path, row_idx, class1_name, class2_name, num_features=10)
        if result:
            results.append(result)

    # Summary comparison
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY COMPARISON")
        print(f"{'='*80}")

        print(
            f"{'Row':<5} {'Predicted':<15} {'True':<15} {f'{class1_name}':<12} {f'{class2_name}':<12} {'Difference':<12}"
        )
        print(f"{'-'*80}")

        for result in results:
            print(
                f"{result['row_index']:<5} {result['predicted_class']:<15} {result['true_class']:<15} {result['class1_prob']:<12.4f} {result['class2_prob']:<12.4f} {result['prob_difference']:<12.4f}"
            )

    return results


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare probabilities between two specific classes")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("--row_index", type=int, help="Index of the row to analyze")
    parser.add_argument("--row_indices", type=str, help="Comma-separated list of row indices to compare")
    parser.add_argument("--class1", type=str, required=True, help="First class name (e.g., HEADING)")
    parser.add_argument("--class2", type=str, required=True, help="Second class name (e.g., HEADING_CONT)")
    parser.add_argument("--num_features", type=int, default=15, help="Number of top features to show")

    args = parser.parse_args()

    if args.row_index is not None:
        compare_class_probabilities(
            args.model_path, args.data_path, args.row_index, args.class1, args.class2, args.num_features
        )
    elif args.row_indices is not None:
        row_indices = [int(x.strip()) for x in args.row_indices.split(',')]
        compare_multiple_rows(args.model_path, args.data_path, row_indices, args.class1, args.class2)
    else:
        print("Please specify either --row_index or --row_indices")


if __name__ == "__main__":
    main()
