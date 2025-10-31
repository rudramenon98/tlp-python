import logging
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

log = logging.getLogger(__name__)


class TextType(Enum):
    HEADER = 0
    BODY = 1
    FOOTER = 2


def identify_header_footer_regressor(pdf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify headers and footers using a regressor to predict y-coordinates and classify based on predicted values.

    This method trains a regressor on text block features to predict y-coordinates, then classifies based on:
    - Low predicted y-values: HEADER
    - High predicted y-values: FOOTER
    - Medium predicted y-values: BODY

    Args:
        pdf_df: DataFrame with text blocks containing y-coordinate information

    Returns:
        pd.DataFrame: Original dataframe with added 'header_footer_type' column
    """
    df = pdf_df.copy()

    # Ensure we have the required y-coordinate columns
    required_cols = [
        "y0",
        "y1",
        "block_y0",
        "block_y1",
        "line_y0",
        "line_y1",
        "PageNumber",
        "page_height",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Initialize the classification column
    df["header_footer_type"] = TextType.BODY.value

    # Define features to use for regression
    feature_columns = [
        # Line Spacing Features
        "left_indent_binned",
        "right_space_binned",
        "space_above_binned",
        "space_below_binned",
        # Font Features
        "font_size_binned",
        "is_bold",
        "is_italic",
        "bold_changed",
        "font_size_changed",
        "italic_changed",
        "font_color_changed",
        "font_family_changed",
        # Text Features
        "num_chars_binned",
        "num_words_binned",
        "is_all_caps",
        "is_title_case",
        "more_space_below",
        # Position features
        "PageNumber",
    ]

    # Check which features are available
    available_features = [col for col in feature_columns if col in df.columns]
    missing_features = [col for col in feature_columns if col not in df.columns]

    if missing_features:
        log.warning(f"Missing feature columns: {missing_features}")

    if not available_features:
        raise ValueError("No feature columns available for regression")

    log.debug(
        f"Using {len(available_features)} features for regression: {available_features}"
    )

    # Prepare feature matrix
    X = df[available_features].copy()

    # Handle categorical features by encoding them
    categorical_features = ["font_family_changed", "font_color_changed"]
    for feature in categorical_features:
        if feature in X.columns:
            # Simple label encoding for categorical features
            X[feature] = pd.Categorical(X[feature]).codes

    # Handle any NaN values
    X = X.fillna(0)

    # Target variable: normalized y-coordinate within each page
    y_targets = []
    for page_num in df["PageNumber"].unique():
        page_mask = df["PageNumber"] == page_num
        page_height = df[page_mask]["page_height"].iloc[0]
        # Normalize y-coordinates to 0-1 within each page
        normalized_y = df.loc[page_mask, "block_y0"] / page_height
        y_targets.extend(normalized_y.values)

    y = np.array(y_targets)

    log.debug(f"Feature matrix shape: {X.shape}")
    log.debug(f"Target variable shape: {y.shape}")

    # Train Random Forest Regressor
    rf_regressor = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )

    # Fit the model
    rf_regressor.fit(X, y)

    # Predict y-coordinates
    y_pred = rf_regressor.predict(X)

    # Calculate performance metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    log.debug(f"Regression performance - MSE: {mse:.4f}, R²: {r2:.4f}")

    # Get feature importance
    feature_importance = pd.DataFrame(
        {"feature": available_features, "importance": rf_regressor.feature_importances_}
    ).sort_values("importance", ascending=False)

    log.debug(f"Top 5 most important features:")
    for _, row in feature_importance.head().iterrows():
        log.debug(f"  {row['feature']}: {row['importance']:.4f}")

    # Classify based on predicted y-coordinates
    # Use quantiles to determine thresholds
    q33 = np.percentile(y_pred, 33)
    q67 = np.percentile(y_pred, 67)

    log.debug(f"Classification thresholds - Q33: {q33:.4f}, Q67: {q67:.4f}")

    classifications = []
    for pred_y in y_pred:
        if pred_y <= q33:
            classifications.append(TextType.HEADER.value)
        elif pred_y >= q67:
            classifications.append(TextType.FOOTER.value)
        else:
            classifications.append(TextType.BODY.value)

    df["header_footer_type"] = classifications
    df["predicted_y"] = y_pred

    log.debug(f"Document-wide regression-based classification completed")
    log.debug(f"Num Headers: {np.bincount(classifications)[TextType.HEADER.value]}")
    log.debug(f"Num Bodies: {np.bincount(classifications)[TextType.BODY.value]}")
    log.debug(f"Num Footers: {np.bincount(classifications)[TextType.FOOTER.value]}")
    log.debug(f"Classifications: {np.bincount(classifications)}")

    return df


def main():
    from ai_doc_parser import DATA_DIR

    data_dir = DATA_DIR / "documents" / "EASA" / "computed_features"
    file_name = "Easy Access Rules for Aerodromes _PDF_"
    output_dir = DATA_DIR / "documents" / "EASA" / "header_footer_classified"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{file_name}.csv"
    print(file_name)

    pdf_path = data_dir / f"{file_name}.csv"
    pdf_df = pd.read_csv(pdf_path)

    # Test the regressor approach
    result_df = identify_header_footer_regressor(pdf_df)
    result_df.to_csv(output_path, index=False)

    # Print classification summary
    print("\nRegression-based Classification Summary:")
    print(result_df["header_footer_type"].value_counts())

    # Print type names for better readability
    header_footer_type_names = {0: "HEADER", 1: "BODY", 2: "FOOTER"}
    result_df["header_footer_type_name"] = result_df["header_footer_type"].map(
        header_footer_type_names
    )
    print("\nClassification Summary by Type:")
    print(result_df["header_footer_type_name"].value_counts())


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
