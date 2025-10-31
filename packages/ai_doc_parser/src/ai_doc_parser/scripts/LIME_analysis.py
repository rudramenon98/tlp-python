"""
LIME Analysis Module

This module provides LIME (Local Interpretable Model-agnostic Explanations) analysis
for understanding individual predictions in the AI PDF parser.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Add the src directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

import joblib
from ai_doc_parser.text_class import CLASS_MAP_INV, TextClass
from ai_doc_parser.training.classifier_trainer import FEATURE_COLUMNS

try:
    from lime import lime_tabular

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")


def lime_analysis(
    model: RandomForestClassifier,
    features_df: pd.DataFrame,
    pdf_idx: int,
    class_1: TextClass,
    class_2: TextClass,
) -> None:
    """
    Perform LIME analysis on a specific row to understand feature contributions
    to the decision between two classes.

    Parameters:
    -----------
    model : RandomForestClassifier
        The trained RandomForest model
    features_df : pd.DataFrame
        DataFrame containing features with 'pdf_idx' column
    pdf_idx : int
        The pdf_idx value to filter for the specific row
    class_1 : TextClass
        First class for comparison
    class_2 : TextClass
        Second class for comparison

    Returns:
    --------
    None
        Prints a table showing feature values and their effects on the decision
    """

    if not LIME_AVAILABLE:
        print("Error: LIME is not available. Please install with: pip install lime")
        return

    # Filter the dataframe for the specific pdf_idx
    filtered_df = features_df[features_df["pdf_idx"] == pdf_idx]

    if len(filtered_df) == 0:
        print(f"Error: No rows found with pdf_idx = {pdf_idx}")
        return

    if len(filtered_df) > 1:
        print(
            f"Warning: Multiple rows found with pdf_idx = {pdf_idx}. Using the first one."
        )
        filtered_df = filtered_df.iloc[:1]

    # Get the row data
    row_data = filtered_df.iloc[0]

    # Identify feature columns (exclude non-feature columns)
    feature_columns = FEATURE_COLUMNS
    print(f"Using {len(feature_columns)} numeric features for LIME analysis")

    # Prepare feature data for LIME - ensure all data is numeric
    X_features = features_df[feature_columns].copy()
    sample_features = row_data[feature_columns].copy()

    # Convert to numeric, handling any remaining non-numeric values
    for col in feature_columns:
        X_features[col] = pd.to_numeric(X_features[col], errors="coerce")
        sample_features[col] = pd.to_numeric(sample_features[col], errors="coerce")

    # Fill any NaN values with 0 (this is a simple approach)
    X_features = X_features.fillna(0)
    sample_features = sample_features.fillna(0)

    # Convert to numpy arrays
    X_features = X_features.values
    sample_features = sample_features.values.reshape(1, -1)

    # Get class names
    class_names = [
        TextClass(CLASS_MAP_INV[i]).name for i in sorted(CLASS_MAP_INV.keys())
    ]

    # Get probabilities for the two classes of interest
    class_1_idx = None
    class_2_idx = None

    for idx, class_val in CLASS_MAP_INV.items():
        if class_val == class_1:
            class_1_idx = idx
        elif class_val == class_2:
            class_2_idx = idx

    if class_1_idx is None or class_2_idx is None:
        print(
            f"Error: Could not find class indices for {class_1.name} and {class_2.name}"
        )
        return

    # Create LIME explainer
    lime_explainer = lime_tabular.LimeTabularExplainer(
        X_features,
        feature_names=feature_columns,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    # Generate explanation using the standard approach
    explanation = lime_explainer.explain_instance(
        sample_features[0], model.predict_proba, num_features=len(feature_columns)
    )

    # Get the explanation as a list
    exp_list = explanation.as_list()

    # Get prediction
    prediction = model.predict(sample_features)[0]
    prediction_proba = model.predict_proba(sample_features)[0]
    predicted_class = (
        TextClass(CLASS_MAP_INV[prediction]).name
        if prediction in CLASS_MAP_INV
        else f"Class_{prediction}"
    )

    prob_class_1 = prediction_proba[class_1_idx]
    prob_class_2 = prediction_proba[class_2_idx]

    class_1_name = class_1.name.split(" ")[0]
    class_2_name = class_2.name.split(" ")[0]

    # Print header information
    print("=" * 80)
    print("LIME ANALYSIS RESULTS")
    print("=" * 80)
    print(f"PDF Index: {pdf_idx}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Probability for {class_1_name}: {prob_class_1:.4f}")
    print(f"Probability for {class_2_name}: {prob_class_2:.4f}")
    print(f"Decision: {class_1_name if prob_class_1 > prob_class_2 else class_2_name}")
    print("=" * 80)

    # Create a comprehensive table
    print(f"Class 1: {class_1_name}")
    print(f"Class 2: {class_2_name}")
    print(
        f"\n{'Feature Name':<50} {'Value':<12} {'LIME Weight':<12} {'Effect':<15} {'Class 1 Contrib':<15} {'Class 2 Contrib':<15}"
    )
    print("-" * 120)

    # Sort features by absolute LIME weight
    exp_list_sorted = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)

    for feature_name, lime_weight in exp_list_sorted:
        feature_name = feature_name.split(" ")[0]
        # Get the actual feature value
        feature_idx = (
            feature_columns.index(feature_name)
            if feature_name in feature_columns
            else -1
        )
        feature_value = sample_features[0][feature_idx] if feature_idx >= 0 else np.nan

        # For the two-class comparison, we need to interpret the LIME weight in context
        # Positive weight means it increases the probability of the predicted class
        # We need to determine how this affects the comparison between our two classes

        # Calculate the probability difference between the two classes
        prob_class_1 - prob_class_2

        # Determine the effect based on the LIME weight and the probability difference
        if lime_weight > 0:
            # Positive weight increases the probability of the predicted class
            if prob_class_1 > prob_class_2:
                # Class 1 is predicted, positive weight favors class 1
                effect = f"Favors {class_1_name}"
                class_1_contrib = lime_weight
                class_2_contrib = -lime_weight
            else:
                # Class 2 is predicted, positive weight favors class 2
                effect = f"Favors {class_2_name}"
                class_1_contrib = -lime_weight
                class_2_contrib = lime_weight
        else:
            # Negative weight decreases the probability of the predicted class
            if prob_class_1 > prob_class_2:
                # Class 1 is predicted, negative weight opposes class 1
                effect = f"Opposes {class_1_name}"
                class_1_contrib = lime_weight
                class_2_contrib = -lime_weight
            else:
                # Class 2 is predicted, negative weight opposes class 2
                effect = f"Opposes {class_2_name}"
                class_1_contrib = -lime_weight
                class_2_contrib = lime_weight

        print(
            f"{feature_name:<50} {feature_value:<12.4f} {lime_weight:<12.4f} {effect:<15} {class_1_contrib:<15.4f} {class_2_contrib:<15.4f}"
        )

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    positive_weights = [weight for _, weight in exp_list_sorted if weight > 0]
    negative_weights = [weight for _, weight in exp_list_sorted if weight < 0]

    print(f"Total features analyzed: {len(exp_list_sorted)}")
    print(f"Features with positive LIME weights: {len(positive_weights)}")
    print(f"Features with negative LIME weights: {len(negative_weights)}")
    print(
        f"Strongest positive influence: {max(exp_list_sorted, key=lambda x: x[1])[0]} ({max(exp_list_sorted, key=lambda x: x[1])[1]:.4f})"
    )
    print(
        f"Strongest negative influence: {min(exp_list_sorted, key=lambda x: x[1])[0]} ({min(exp_list_sorted, key=lambda x: x[1])[1]:.4f})"
    )

    # Feature value summary
    print(f"\nTop 10 Most Influential Features:")
    print(f"{'Feature':<50} {'Value':<12} {'LIME Weight':<12} {'Effect':<15}")
    print("-" * 90)

    for feature_name, lime_weight in exp_list_sorted[:10]:  # Top 10 features
        feature_name = feature_name.split(" ")[0]
        feature_idx = (
            feature_columns.index(feature_name)
            if feature_name in feature_columns
            else -1
        )
        if feature_idx >= 0:
            feature_value = sample_features[0][feature_idx]
            # Determine effect
            if lime_weight > 0:
                effect = f"Favors {class_1_name if prob_class_1 > prob_class_2 else class_2_name}"
            else:
                effect = f"Opposes {class_1_name if prob_class_1 > prob_class_2 else class_2_name}"
            print(
                f"{feature_name:<50} {feature_value:<12.4f} {lime_weight:<12.4f} {effect:<15}"
            )

    print("\n" + "=" * 80)
    print("LIME Analysis Complete")
    print("=" * 80)


def main():
    # Use absolute paths for testing
    model_path = Path(
        "C:/Users/r123m/Documents/enginius/source/ai-pdf-parser/data/documents/models/RandomForestClassifier.sav"
    )
    features_path = Path(
        "C:/Users/r123m/Documents/enginius/source/ai-pdf-parser/data/documents/MDR/computed_features/CELEX_32017R0745_EN_TXT.csv"
    )
    pdf_idx = 152
    class_1 = TextClass.PARAGRAPH_CONT
    class_2 = TextClass.TOC
    logging.basicConfig(level=logging.DEBUG)

    model = joblib.load(model_path)
    features_df = pd.read_csv(features_path)
    lime_analysis(model, features_df, pdf_idx, class_1, class_2)


if __name__ == "__main__":
    main()
