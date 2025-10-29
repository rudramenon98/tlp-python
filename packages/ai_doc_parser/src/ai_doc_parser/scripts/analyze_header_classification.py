"""
Script to analyze why a line was classified as HEADER_CONT instead of HEADER.

This helps understand the specific patterns that lead to header continuation classification.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from ai_doc_parser.training.classifier_trainer import load_model, prepare_df_for_model
from ai_doc_parser.text_class import TextClass, CLASS_MAP_INV


def analyze_header_classification(model_path: str, features_path: str, row_index: int):
    """
    Analyze why a specific row was classified as HEADER_CONT instead of HEADER.
    """
    # Load model and data
    model = load_model(model_path)
    df = pd.read_csv(features_path)

    # Prepare data
    df_processed, feature_columns = prepare_df_for_model(df, add_shifted_features=True, verbose=False)
    X = df_processed[feature_columns]
    y_true = df_processed['LabelledClass'] if 'LabelledClass' in df_processed.columns else None

    # Get the specific row and surrounding context
    if row_index >= len(X):
        print(f"Row index {row_index} out of range. Data has {len(X)} rows.")
        return

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

    print(f"\n{'='*80}")
    print(f"HEADER CLASSIFICATION ANALYSIS FOR ROW {row_index}")
    print(f"{'='*80}")
    print(f"Predicted Class: {pred_class_name} (Class {prediction})")
    print(f"True Class: {true_class_name} (Class {true_label})")
    print(f"Prediction Confidence: {max(prediction_proba):.3f}")

    # Get probabilities for all classes
    class_names = [TextClass(CLASS_MAP_INV[i]).name for i in sorted(CLASS_MAP_INV.keys())]
    print(f"\nClass Probabilities:")
    for i, class_name in enumerate(class_names):
        if i < len(prediction_proba):
            print(f"  {class_name}: {prediction_proba[i]:.3f}")

    # Focus on HEADER vs HEADER_CONT comparison
    header_class = None
    header_cont_class = None

    for i, class_name in enumerate(class_names):
        if class_name == "HEADING":
            header_class = i
        elif class_name == "HEADING_CONT":
            header_cont_class = i

    if header_class is not None and header_cont_class is not None:
        header_prob = prediction_proba[header_class]
        header_cont_prob = prediction_proba[header_cont_class]

        print(f"\nHEADER vs HEADER_CONT Analysis:")
        print(f"  HEADER probability: {header_prob:.3f}")
        print(f"  HEADER_CONT probability: {header_cont_prob:.3f}")
        print(f"  Difference: {header_cont_prob - header_prob:.3f}")

    # Analyze key features that distinguish HEADER from HEADER_CONT
    print(f"\n{'='*80}")
    print("KEY FEATURES ANALYSIS")
    print(f"{'='*80}")

    feature_values = sample.iloc[0]
    feature_importance = model.feature_importances_

    # Focus on features that are likely to distinguish headers
    header_related_features = [
        'is_bold',
        'is_italic',
        'font_size_bin',
        'left_indent_bin',
        'text_length_bin',
        'ends_with_period',
        'ends_with_letter',
        'is_title_case',
        'all_caps',
        'number_dots',
    ]

    # Add shifted features
    shifted_features = []
    for feature in header_related_features:
        for shift in [-1, 1]:
            shifted_features.append(f"{feature}_shift_{shift}")

    all_header_features = header_related_features + shifted_features

    print(f"{'Feature':<35} {'Value':<10} {'Importance':<12}")
    print(f"{'-'*80}")

    all_header_features = sorted(
        all_header_features, key=lambda x: feature_importance[feature_columns.index(x)], reverse=True
    )

    for feature in all_header_features:
        if feature in feature_columns:
            idx = feature_columns.index(feature)
            value = feature_values.iloc[idx]
            importance = feature_importance[idx]

            print(f"{feature:<35} {value:<10.3f} {importance:<12.4f}")

    # Context analysis - look at surrounding rows
    print(f"\n{'='*80}")
    print("CONTEXT ANALYSIS (Surrounding Rows)")
    print(f"{'='*80}")

    context_range = 3  # Look at 3 rows before and after
    start_idx = max(0, row_index - context_range)
    end_idx = min(len(X), row_index + context_range + 1)

    print(f"Analyzing rows {start_idx} to {end_idx-1} (target row: {row_index})")
    print(f"{'Row':<5} {'Predicted':<15} {'True':<15} {'Confidence':<12} {'Key Features':<30}")
    print(f"{'-'*80}")

    for i in range(start_idx, end_idx):
        if i < len(X):
            context_sample = X.iloc[i : i + 1]
            context_pred = model.predict(context_sample)[0]
            context_proba = model.predict_proba(context_sample)[0]
            context_true = y_true.iloc[i] if y_true is not None else None

            pred_name = (
                TextClass(CLASS_MAP_INV[context_pred]).name
                if context_pred in CLASS_MAP_INV
                else f"Class_{context_pred}"
            )
            true_name = (
                TextClass(CLASS_MAP_INV[context_true]).name
                if context_true in CLASS_MAP_INV
                else f"Class_{context_true}"
            )

            # Get key features for this row
            key_features = []
            if context_sample.iloc[0]['is_bold'] > 0.5:
                key_features.append("Bold")
            if context_sample.iloc[0]['left_indent_bin'] < 1.0:
                key_features.append("LowIndent")
            if context_sample.iloc[0]['font_size_bin'] > 4.0:
                key_features.append("LargeFont")

            key_features_str = ",".join(key_features[:3])  # Limit to 3 features

            marker = " <-- TARGET" if i == row_index else ""
            print(f"{i:<5} {pred_name:<15} {true_name:<15} {max(context_proba):<12.3f} {key_features_str:<30}{marker}")

    # LIME-style analysis for header-specific features
    print(f"\n{'='*80}")
    print("HEADER-SPECIFIC FEATURE ANALYSIS")
    print(f"{'='*80}")

    # Analyze what makes this look like HEADER_CONT vs HEADER
    header_cont_indicators = []
    header_indicators = []

    # Check specific patterns
    if feature_values['is_bold'] > 0.5:
        header_cont_indicators.append("Bold text (HEADER_CONT pattern)")
    else:
        header_indicators.append("Not bold (HEADER pattern)")

    if feature_values['left_indent_bin'] <= 1.0:
        header_cont_indicators.append("Low indentation (continuation pattern)")
    else:
        header_indicators.append("Higher indentation (standalone pattern)")

    if 'text_length_bin_shift_1' in feature_columns:
        next_line_length = feature_values['text_length_bin_shift_1']
        if next_line_length <= 1.0:
            header_cont_indicators.append("Short next line (continuation)")
        else:
            header_indicators.append("Long next line (standalone)")

    print("Features suggesting HEADER_CONT:")
    for indicator in header_cont_indicators:
        print(f"  ✓ {indicator}")

    print("\nFeatures suggesting HEADER:")
    for indicator in header_indicators:
        print(f"  ✓ {indicator}")

    # LIME analysis for header-specific features
    print(f"\n{'='*80}")
    print("LIME EXPLANATION FOR HEADER CLASSIFICATION")
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

        # Generate explanation
        explanation = lime_explainer.explain_instance(sample.values[0], model.predict_proba, num_features=15)

        print("LIME Explanation (Top 15 features):")
        print(f"{'Feature':<35} {'Weight':<10} ")
        print(f"{'-'*65}")
        max_feature_len = max(len(feature) for feature in feature_columns)

        exp_list = explanation.as_list()
        # sort by weight
        exp_list = sorted(exp_list, key=lambda x: x[1], reverse=True)
        for feature, weight in exp_list:
            print(f"{feature:<{max_feature_len + 8}} {weight:<10.3f}")

        # Save LIME explanation
        Path("header_analysis_output").mkdir(exist_ok=True)
        explanation.save_to_file(f"header_analysis_output/lime_explanation_row_{row_index}.html")
        print(f"\nLIME explanation saved to: header_analysis_output/lime_explanation_row_{row_index}.html")

    except ImportError:
        print("LIME not available. Install with: pip install lime")
    except Exception as e:
        print(f"LIME analysis failed: {e}")

    print(f"\nAnalysis completed!")


def main():
    """Main function."""
    import argparse

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

    analyze_header_classification(model_path, features_path, row_index)


if __name__ == "__main__":
    main()
