import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from ai_doc_parser.common import load_model
from ai_doc_parser.training.parsers.base_parser import TextClass
from sklearn.ensemble import RandomForestClassifier


def predict_single(
    model: RandomForestClassifier, features: Union[List[float], pd.Series]
) -> int:
    """Make a prediction on a single sample."""
    return model.predict([features])[0]


def predict_proba_single(
    model: RandomForestClassifier, features: Union[List[float], pd.Series]
) -> np.ndarray:
    """Get prediction probabilities for a single sample."""
    return model.predict_proba([features])[0]


def prepare_features_for_inference(
    df: pd.DataFrame, feature_columns: List[str]
) -> tuple[pd.DataFrame, List[str]]:
    """
    Prepare features for inference by ensuring all required columns exist
    and handling missing values.

    Args:
        df: Input dataframe with features
        feature_columns: List of feature column names to use

    Returns:
        Tuple of (prepared_features, available_columns)
    """
    # Create text-based features if text column exists
    if "text" in df.columns:
        # Create text length feature
        df["text_length"] = df["text"].str.len()
        feature_columns.append("text_length")

    # Ensure all feature columns exist
    available_columns = [col for col in feature_columns if col in df.columns]
    missing_columns = [col for col in feature_columns if col not in df.columns]

    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")

    print(f"Using {len(available_columns)} feature columns: {available_columns}")

    # Prepare features
    x = df[available_columns].copy()

    # Handle missing values in features
    # Fill numeric NaNs with 0
    x = x.fillna(0)

    return x, available_columns


def get_feature_columns() -> List[str]:
    """Get the standard feature columns used for classification."""
    return [
        "x0",
        "y0",
        "x1",
        "y1",
        "block_x0",
        "block_y0",
        "block_x1",
        "block_y1",
        "line_x0",
        "line_y0",
        "line_x1",
        "line_y1",
        "page_number",
        "page_height",
        "page_width",
        "major_font_size",
        "ncols",
        "is_table",
    ]


def classify_dataframe(
    df: pd.DataFrame,
    model: RandomForestClassifier,
    removed_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Classify a dataframe using a trained model.

    Args:
        df: DataFrame with features to classify
        model: Trained RandomForestClassifier model
        removed_columns: List of columns to remove from features

    Returns:
        DataFrame with classification results added
    """
    # Get feature columns
    feature_columns = get_feature_columns()

    # Remove specified columns from features
    if removed_columns:
        feature_columns = [col for col in feature_columns if col not in removed_columns]

    # Prepare features for inference
    x, available_columns = prepare_features_for_inference(df, feature_columns)

    print("Making predictions...")
    predictions = model.predict(x)
    prediction_probas = model.predict_proba(x)

    # Add predictions to dataframe
    df["predicted_class"] = predictions
    df["predicted_class_name"] = [TextClass(pred).name for pred in predictions]

    # Add prediction probabilities for each class
    class_names = [TextClass(i).name for i in range(len(model.classes_))]
    for i, class_name in enumerate(class_names):
        df[f"prob_{class_name.lower()}"] = prediction_probas[:, i]

    return df


def save_classification_results(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Save classification results to CSV file.

    Args:
        df: DataFrame with classification results
        output_path: Path to save results

    Returns:
        Path to the saved file
    """

    print(f"Saving classified results to {output_path}...")
    df.to_csv(output_path, index=False)

    return output_path


def print_classification_summary(
    df: pd.DataFrame, available_columns: List[str]
) -> None:
    """
    Print a summary of classification results.

    Args:
        df: DataFrame with classification results
        available_columns: List of feature columns used
    """
    print("\n" + "=" * 50)
    print("CLASSIFICATION SUMMARY")
    print("=" * 50)
    print(f"Total lines processed: {len(df)}")
    print(f"Feature columns used: {len(available_columns)}")

    # Class distribution
    class_counts = df["predicted_class"].value_counts().sort_index()
    print("\nPredicted Class Distribution:")
    for class_id, count in class_counts.items():
        class_name = TextClass(class_id).name
        percentage = (count / len(df)) * 100
        print(f"  {class_name} (Class {class_id}): {count} lines ({percentage:.1f}%)")

    print("=" * 50)


def run_inference(
    model: RandomForestClassifier,
    features_df: pd.DataFrame,
    removed_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run inference on a features CSV file using a trained model.

    Args:
        model_path: Path to the trained model file
        features_csv_path: Path to the features CSV file
        removed_columns: List of columns to remove from features (default: None)
    """

    # Classify the dataframe
    df = classify_dataframe(features_df, model, removed_columns)

    # Print summary
    feature_columns = get_feature_columns()
    if removed_columns:
        feature_columns = [col for col in feature_columns if col not in removed_columns]
    _, available_columns = prepare_features_for_inference(df, feature_columns)

    print_classification_summary(df, available_columns)

    return df


def main():
    """Main function to run inference."""
    data_dir = Path(__file__).parents[3] / "data"

    # Paths
    model_path = data_dir / "models" / "RandomForestClassifier.sav"
    features_csv_path = (
        data_dir
        / "xml_cfr"
        / "computed_features"
        / "CFR-2024-title14-vol2_features.csv"
    )
    features_df = pd.read_csv(features_csv_path)
    print(f"Features dataframe loaded from {features_csv_path}")
    output_dir = data_dir / "xml_cfr" / "classified_pdf"

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        exit(1)

    if not os.path.exists(features_csv_path):
        print(f"Error: Features CSV file not found at {features_csv_path}")
        exit(1)

    print("Starting inference...")
    print("=" * 50)

    model = load_model(model_path)
    print(f"Model loaded from {model_path}")

    # Run inference
    df = run_inference(
        model=model,
        features_df=features_df,
        removed_columns=["page_number"],  # Same as in training
    )

    output_name = features_csv_path.name.replace("features", "classified")
    output_path = output_dir / output_name
    save_classification_results(df, output_path)
    print(f"Classification results saved to {output_path}")
    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
