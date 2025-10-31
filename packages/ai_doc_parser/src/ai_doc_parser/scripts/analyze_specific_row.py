"""
Script to analyze what features are driving a specific row's classification.

This script shows how to use LIME and SHAP to understand individual predictions.
"""

import sys
from pathlib import Path

import pandas as pd

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from ai_doc_parser.text_class import CLASS_MAP_INV, TextClass
from ai_doc_parser.tools.model_interpretability import ModelInterpretabilityAnalyzer
from ai_doc_parser.training.classifier_trainer import load_model, prepare_df_for_model


def analyze_specific_row(model_path: str, data_path: str, row_index: int, output_dir: str = "row_analysis"):
    """
    Analyze what features are driving a specific row's classification.

    Args:
        model_path: Path to the trained model
        data_path: Path to the data CSV
        row_index: Index of the row to analyze
        output_dir: Directory to save results
    """
    print(f"Analyzing row {row_index}...")

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

    # Get prediction
    print(sample)
    prediction = model.predict(sample)[0]
    prediction_proba = model.predict_proba(sample)[0]

    # Get class names
    class_names = [TextClass(CLASS_MAP_INV[i]).name for i in sorted(CLASS_MAP_INV.keys())]
    pred_class_name = (
        TextClass(CLASS_MAP_INV[prediction]).name if prediction in CLASS_MAP_INV else f"Class_{prediction}"
    )
    true_class_name = (
        TextClass(CLASS_MAP_INV[true_label]).name if true_label in CLASS_MAP_INV else f"Class_{true_label}"
    )

    print(f"\n{'='*60}")
    print(f"ANALYSIS FOR ROW {row_index}")
    print(f"{'='*60}")
    print(f"Predicted Class: {pred_class_name} (Class {prediction})")
    print(f"Prediction Confidence: {max(prediction_proba):.3f}")
    if true_label is not None:
        print(f"True Class: {true_class_name} (Class {true_label})")
        print(f"Correct: {'Yes' if prediction == true_label else 'No'}")

    # Initialize analyzer
    ModelInterpretabilityAnalyzer(model, feature_columns, class_names)

    # Method 1: Feature Importance Analysis for this specific row
    print(f"\n{'='*60}")
    print("METHOD 1: FEATURE VALUES & IMPORTANCE")
    print(f"{'='*60}")

    feature_values = sample.iloc[0]
    feature_importance = model.feature_importances_

    # Create feature contribution dataframe
    feature_contrib = pd.DataFrame(
        {'feature': feature_columns, 'value': feature_values.values, 'importance': feature_importance}
    )

    # Sort by importance and show top 15
    top_features = feature_contrib.sort_values('importance', ascending=False).head(15)

    print(f"{'Feature':<30} {'Value':<10} {'Importance':<12} {'Contribution':<12}")
    print(f"{'-'*70}")
    for _, row in top_features.iterrows():
        # Contribution is value * importance (simplified)
        contribution = row['value'] * row['importance']
        print(f"{row['feature']:<30} {row['value']:<10.3f} {row['importance']:<12.4f} {contribution:<12.4f}")

    # Method 2: LIME Analysis (if available)
    print(f"\n{'='*60}")
    print("METHOD 2: LIME EXPLANATION")
    print(f"{'='*60}")

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

        # Generate explanation
        explanation = lime_explainer.explain_instance(sample.values[0], model.predict_proba, num_features=15)

        print("LIME Explanation (Top 15 features):")
        print(f"{'Feature':<30} {'Weight':<10} {'Effect':<15}")
        print(f"{'-'*60}")

        max_feature_len = max(len(feature) for feature in feature_columns)
        exp_list = explanation.as_list()
        for feature, weight in exp_list:
            effect = "increases" if weight > 0 else "decreases"
            print(f"{feature:<{max_feature_len}} {weight:<10.3f} {effect:<15}")

        # Save LIME explanation
        Path(output_dir).mkdir(exist_ok=True)
        explanation.save_to_file(f"{output_dir}/lime_explanation_row_{row_index}.html")
        print(f"\nLIME explanation saved to: {output_dir}/lime_explanation_row_{row_index}.html")

    except ImportError:
        print("LIME not available. Install with: pip install lime")
    except Exception as e:
        print(f"LIME analysis failed: {e}")

    # Method 3: SHAP Analysis (if available)
    print(f"\n{'='*60}")
    print("METHOD 3: SHAP EXPLANATION")
    print(f"{'='*60}")

    try:
        import shap

        # Create SHAP explainer
        shap_explainer = shap.TreeExplainer(model)
        shap_values = shap_explainer.shap_values(sample)

        # Handle multi-class case
        if isinstance(shap_values, list):
            # Use the predicted class
            class_shap_values = shap_values[prediction]
            expected_value = shap_explainer.expected_value[prediction]
        else:
            class_shap_values = shap_values[0]
            expected_value = shap_explainer.expected_value

        print(f"SHAP Explanation for class {pred_class_name}:")
        print(f"Expected value (baseline): {expected_value:.3f}")
        print(f"Prediction: {expected_value + class_shap_values.sum():.3f}")
        print(f"\nFeature contributions:")
        print(f"{'Feature':<30} {'Value':<10} {'SHAP Value':<12} {'Effect':<15}")
        print(f"{'-'*75}")

        # Sort by absolute SHAP value
        shap_contrib = pd.DataFrame(
            {'feature': feature_columns, 'value': sample.values[0], 'shap_value': class_shap_values}
        ).sort_values('shap_value', key=abs, ascending=False)

        for _, row in shap_contrib.head(15).iterrows():
            effect = "increases" if row['shap_value'] > 0 else "decreases"
            print(f"{row['feature']:<30} {row['value']:<10.3f} {row['shap_value']:<12.3f} {effect:<15}")

        # Create SHAP waterfall plot
        Path(output_dir).mkdir(exist_ok=True)
        shap.waterfall_plot(expected_value, class_shap_values, sample.iloc[0], show=False, max_display=15)
        import matplotlib.pyplot as plt

        plt.savefig(f"{output_dir}/shap_waterfall_row_{row_index}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSHAP waterfall plot saved to: {output_dir}/shap_waterfall_row_{row_index}.png")

    except ImportError:
        print("SHAP not available. Install with: pip install shap")
    except Exception as e:
        print(f"SHAP analysis failed: {e}")

    # Method 4: Decision Path Analysis
    print(f"\n{'='*60}")
    print("METHOD 4: DECISION PATH ANALYSIS")
    print(f"{'='*60}")

    # Get the decision path for this sample
    decision_paths = model.decision_path(sample)

    print("Decision path through the trees:")
    print("(This shows which features were used in the decision process)")

    # Analyze the decision path
    feature_usage = {}
    for tree_idx, tree in enumerate(model.estimators_):
        path = decision_paths[0, tree_idx]
        # Get the features used in this tree's decision path
        tree_features = tree.tree_.feature[path.nonzero()[1]]
        for feature_idx in tree_features:
            if feature_idx >= 0:  # Skip leaf nodes
                feature_name = feature_columns[feature_idx]
                feature_usage[feature_name] = feature_usage.get(feature_name, 0) + 1

    # Sort by usage frequency
    sorted_features = sorted(feature_usage.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Feature':<30} {'Times Used':<12}")
    print(f"{'-'*45}")
    for feature, count in sorted_features[:15]:
        print(f"{feature:<30} {count:<12}")

    print(f"\nAnalysis completed! Check {output_dir} for saved visualizations.")


def main():
    """Main function to analyze a specific row."""

    model_path = (
        r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\models\RandomForestClassifier.sav"
    )
    data_path = Path(r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\FIPS\NIST.FIPS.204.pdf")
    features_path = data_path.parent / "computed_features" / f"{data_path.stem}.csv"
    prepared_features_path = data_path.parent / "prepared_features" / f"{data_path.stem}.csv"
    pdf_idx = 2

    feature_df = pd.read_csv(features_path)
    prepared_df = pd.read_csv(prepared_features_path)
    row_index = feature_df[feature_df['pdf_idx'] == pdf_idx].index[0]
    row = prepared_df.iloc[row_index]
    print(f"Row index: {row_index}")
    print(row)
    analyze_specific_row(model_path, features_path, row_index, output_dir="row_analysis")


if __name__ == "__main__":
    main()
