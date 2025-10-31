import logging
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from ai_doc_parser.text_class import (
    AI_PARSED_CLASSES,
    CLASS_MAP,
    CLASS_MAP_INV,
    TextClass,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

FEATURE_COLUMNS = [
    # Line Spacing Features
    "left_indent_binned",
    "right_space_binned",
    # "space_above_binned",
    # "space_below_binned",
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
    "num_of_dots_binned",
    "num_of_spaces_binned",
    "max_contiguous_dots_binned",
    "max_contiguous_spaces_binned",
    # Text Starting Words Features
    "first_char_isdigit",
    "first_char_bullet",
    "first_char_upper",
    "first_word_compound",
    "starts_with_keyword",
    "first_char_special",
    # Text Ending Words Features
    "last_char_upper",
    "last_char_punctuation",
    "last_word_digit",
    "last_word_roman",
    # Relative Features
    "prev_page_diff",
    "prev_line_space_below_diff",
    "prev_line_left_margin_diff",
    "prev_line_right_space_diff",
    "prev_line_font_size_diff",
    "prev_line_font_family_diff",
    "next_page_diff",
    "next_line_left_margin_diff",
    "next_line_right_space_diff",
    "next_line_font_size_diff",
    "next_line_font_family_diff",
    "more_space_below",
]


def load_model(model_path: str | Path) -> RandomForestClassifier:
    """Load a trained model from disk."""
    return joblib.load(model_path)


def random_forest_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = None,
) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    log.debug("Training Random Forest with %s estimators...", n_estimators)
    log.debug("Training data shape: %s", x_train.shape)
    log.debug("Target classes: %s", sorted([int(i) for i in np.unique(y_train)]))
    log.debug(
        "Target Class Names: %s",
        sorted([TextClass(CLASS_MAP_INV[i]).name for i in np.unique(y_train)]),
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=42, oob_score=True, max_depth=max_depth
    )
    clf.fit(x_train.astype(np.float32), y_train.astype(np.int32))

    log.debug("Random Forest training completed with %s estimators", n_estimators)
    log.debug("Model score on training data: %.4f", clf.score(x_train, y_train))
    return clf


def add_shifted_columns(df: pd.DataFrame, shift_values: List[int]) -> pd.DataFrame:
    """
    Add columns to the dataframe that represent values from shifted rows.

    Args:
        df: Input dataframe
        shift_values: List of shift values (e.g., [-1, 1] for previous and
                     next row)

    Returns:
        pd.DataFrame: Dataframe with original columns plus shifted columns
    """
    # Collect all shifted columns in a dictionary first
    shifted_columns_dict = {}

    for shift in shift_values:
        for column in df.columns:
            # Create shifted column name
            shifted_column_name = f"{column}_shift_{shift}"

            # Shift the column values
            shifted_values = df[column].shift(shift)

            # if shift < 0 and

            # Store in dictionary instead of inserting immediately
            shifted_columns_dict[shifted_column_name] = shifted_values

    # Create DataFrame from all shifted columns at once
    shifted_df = pd.DataFrame(shifted_columns_dict, index=df.index)

    # Concatenate original dataframe with shifted columns
    result_df = pd.concat([df, shifted_df], axis=1)

    # for shift in shift_values:
    #     column_name = f"block_y0_shift_{shift}"
    #     if shift < 0:
    #         # Convert to float to handle NaN values properly
    #         result_df[column_name] = result_df[column_name].astype(float)
    #         result_df.loc[result_df[column_name] > result_df['block_y0'], column_name] = np.nan
    #     elif shift > 0:
    #         # Convert to float to handle NaN values properly
    #         result_df[column_name] = result_df[column_name].astype(float)
    #         result_df.loc[result_df[column_name] < result_df['block_y0'], column_name] = np.nan

    return result_df


def prepare_df_for_model(
    df: pd.DataFrame,
    add_shifted_features: bool = True,
    shift_values: List[int] = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare a dataframe for model training or inference by selecting and processing features.

    This function can be used for both training and inference. For training, it prepares
    the features and returns the processed dataframe along with the feature column names.
    For inference, it ensures the dataframe has all the required features in the correct format.

    Args:
        df: Input dataframe containing the data to be processed
        removed_columns: List of columns to remove from features (default: None)
        add_shifted_features: Whether to add shifted features for context (default: True)
        shift_values: List of shift values for context features (default: [-1, 1])
        verbose: Whether to print progress information (default: True)

    Returns:
        tuple: (processed_df, feature_columns)
            - processed_df: DataFrame with all features prepared for the model
            - feature_columns: List of column names used as features
    """

    # Define the standard feature columns used by the model

    feature_columns = FEATURE_COLUMNS
    # combine _continued classes to one class
    if "LabelledClass" in df.columns:
        df.loc[df["LabelledClass"] == TextClass.BULLET_LIST_CONT, "LabelledClass"] = (
            TextClass.GEN_LIST_CONT
        )
        df.loc[df["LabelledClass"] == TextClass.ENUM_LIST_CONT, "LabelledClass"] = (
            TextClass.GEN_LIST_CONT
        )
        df["SourceClass"] = df["LabelledClass"].map(lambda x: TextClass(x).name)
        df.loc[df["LabelledClass"] == TextClass.ENUM_LIST, "LabelledClass"] = (
            TextClass.PARAGRAPH
        )
        df.loc[df["LabelledClass"] == TextClass.BULLET_LIST, "LabelledClass"] = (
            TextClass.PARAGRAPH
        )
        df.loc[df["LabelledClass"] == TextClass.GEN_LIST_CONT, "LabelledClass"] = (
            TextClass.PARAGRAPH_CONT
        )
    if "ExtractedClass" in df.columns:
        df.loc[df["ExtractedClass"] == TextClass.BULLET_LIST_CONT, "ExtractedClass"] = (
            TextClass.GEN_LIST_CONT
        )
        df.loc[df["ExtractedClass"] == TextClass.ENUM_LIST_CONT, "ExtractedClass"] = (
            TextClass.GEN_LIST_CONT
        )
        # df['ExtractedClassName'] = df['ExtractedClass'].map(lambda x: TextClass(x).name)

    if "LabelledClass" in df.columns:
        # filter out rows that are not in AI_PARSED_CLASSES
        df = df[
            (df["LabelledClass"].isna()) | (df["LabelledClass"].isin(AI_PARSED_CLASSES))
        ]

    if "ExtractedClass" in df.columns:
        # filter out rows that are not in AI_PARSED_CLASSES
        df = df[
            (df["ExtractedClass"].isna())
            | (df["ExtractedClass"].isin(AI_PARSED_CLASSES))
        ]

    # Add text-based features if text column exists
    # if 'text' in df.columns:
    #     # Create text length feature
    #     df['text_length'] = df['text'].str.len()
    #     feature_columns.append('text_length')
    #     if verbose:
    #         print("Added text_length feature")

    # Remove specified columns from features

    # Ensure all feature columns exist
    available_columns = [col for col in feature_columns if col in df.columns]
    missing_columns = [col for col in feature_columns if col not in df.columns]

    if missing_columns:
        log.warning("Missing columns: %s", missing_columns)
        raise ValueError(f"Missing columns: {missing_columns}")

    if verbose:
        log.debug(
            "Using %s feature columns: %s", len(available_columns), available_columns
        )

    if "LabelledClass" in df.columns:
        df = df[["LabelledClass"] + available_columns]
    elif "ExtractedClass" in df.columns:
        df = df[["ExtractedClass"] + available_columns]
    else:
        raise ValueError("No labelled or extracted class column found")
    # Add features of previous and next text block if requested
    # shift_values = []
    if shift_values is None:
        shift_values = [-1, 1]
    if add_shifted_features:
        df = add_shifted_columns(df, shift_values)
        shifted_columns = [
            f"{col}_shift_{shift}"
            for col in available_columns
            for shift in shift_values
        ]
        available_columns.extend(shifted_columns)
        if verbose:
            log.debug("Added %s shifted features", len(shifted_columns))

    # df.to_csv("shifted_labelled_pdf.csv")
    # Prepare features
    # x = df[available_columns].copy()

    if verbose:
        log.debug("Final feature matrix shape: %s", df.shape)

    return df, available_columns


class ModelType(Enum):
    RANDOM_FOREST = "RandomForest"
    XGBOOST = "XGBoost"


def compute_class_centroid(x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """
    Compute the average feature value for each class.

    Args:
        x_train: Training features DataFrame
        y_train: Training labels Series

    Returns:
        pd.DataFrame: DataFrame with features as index and classes as columns,
                     containing the average feature values for each class
    """
    # Combine features and labels
    data = x_train.copy()
    data["class"] = y_train

    # Group by class and compute mean for each feature
    class_centroids = data.groupby("class").mean().round(2)

    # Get class names for column headers
    class_names = class_centroids.index.map(lambda x: TextClass(CLASS_MAP_INV[x]).name)

    # Transpose the DataFrame so features are rows and classes are columns
    class_centroids_transposed = class_centroids.T

    # Rename columns to use class names instead of class numbers
    class_centroids_transposed.columns = class_names

    # Add Feature Name column as the first column
    class_centroids_transposed.insert(
        0, "Feature Name", class_centroids_transposed.index
    )

    return class_centroids_transposed


def train_multiclass(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    model_type: ModelType = ModelType.RANDOM_FOREST,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    Union[RandomForestClassifier, xgboost.XGBClassifier],
    List[str],
    pd.DataFrame,
]:
    """
    Clean function to train a multiclass classifier on labeled CSV data.

    Args:
        csv_path: Path to the labeled CSV file
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        tuple: (x_train, x_test, y_train, y_test, model, feature_columns)
    """

    # Remove rows with missing labels
    df = df.dropna(subset=["LabelledClass"])
    df = df[df["LabelledClass"].notna()]

    df.loc[df["LabelledClass"] == TextClass.BULLET_LIST_CONT, "LabelledClass"] = (
        TextClass.GEN_LIST_CONT
    )
    df.loc[df["LabelledClass"] == TextClass.ENUM_LIST_CONT, "LabelledClass"] = (
        TextClass.GEN_LIST_CONT
    )

    df = df[df["LabelledClass"].isin(AI_PARSED_CLASSES)]

    # Convert LabelledClass to numeric if it's not already
    df["LabelledClass"] = pd.to_numeric(df["LabelledClass"], errors="coerce")

    # Ensure LabelledClass is integer
    df["LabelledClass"] = df["LabelledClass"].astype(int)
    df["LabelledClassName"] = df["LabelledClass"].map(lambda x: TextClass(x).name)

    log.debug("After cleaning: %s rows", len(df))
    original_class_dist = df["LabelledClass"].value_counts().sort_index().to_dict()
    log.debug("Original class distribution: %s", original_class_dist)

    # Balance the dataset: cap each class at max_samples_per_class
    df.index = range(len(df))
    # max_samples_per_class = 20000
    # unique_classes = np.unique(df['LabelledClass'])
    # indices_to_keep = []
    # for unique_class in unique_classes:
    #     class_indices = list(df[df['LabelledClass'] == unique_class].index)
    #     if len(class_indices) <= max_samples_per_class:
    #         indices_to_keep.extend(class_indices)
    #         continue
    #     # shuffle the class indices
    #     np.random.shuffle(class_indices)
    #     # take the first max_samples_per_class indices
    #     class_indices = class_indices[:max_samples_per_class]
    #     # drop the class indices from the dataframe
    #     indices_to_keep.extend(class_indices)

    # df = df.iloc[indices_to_keep]

    # Prepare features usin the shared function
    df, available_columns = prepare_df_for_model(
        df=df, add_shifted_features=True, verbose=True
    )

    # Prepare target variable
    x = df[available_columns]
    y = df["LabelledClass"]

    # Split the data first
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    class_centroid = compute_class_centroid(x_train, y_train)
    # print class centroids as a table
    log.debug("Class Centroid: \n%s", class_centroid.to_string())

    # Apply mapping to both train and test sets
    y_train = np.asarray([CLASS_MAP[y] for y in y_train])
    y_test = np.asarray([CLASS_MAP[y] for y in y_test])

    log.debug("Training set classes: %s", sorted(np.unique(y_train)))
    log.debug("Test set classes: %s", sorted(np.unique(y_test)))

    log.debug("Training set size: %s", len(x_train))
    log.debug("Test set size: %s", len(x_test))
    log.debug("Number of classes: %s", len(y.unique()))
    log.debug("Feature matrix shape: %s", x.shape)
    log.debug("-" * 50)
    log.debug("Starting model training...")
    log.debug("-" * 50)

    # Train the model
    if model_type == ModelType.RANDOM_FOREST:
        log.debug("Training Random Forest model...")
        model = random_forest_model(x_train, y_train, n_estimators=100, max_depth=None)
    elif model_type == ModelType.XGBOOST:
        log.debug("Training XGBoost model...")
        log.debug("Training data shape: %s", x_train.shape)
        log.debug("Target classes: %s", sorted([int(i) for i in np.unique(y_train)]))
        log.debug(
            "Target Class Names: %s",
            sorted([TextClass(CLASS_MAP_INV[i]).name for i in np.unique(y_train)]),
        )

        model = xgboost.XGBClassifier(
            objective="multi:softmax",
            num_class=len(np.unique(y_train)),
            random_state=42,
            verbose=False,
            eval_metric="mlogloss",
        )
        log.debug("XGBoost training started with %s classes", len(np.unique(y_train)))

        model.fit(
            x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_test, y_test)],
        )
        log.debug("XGBoost training completed")
        log.debug("Model score on training data: %.4f", model.score(x_train, y_train))

    y_test = np.asarray([CLASS_MAP_INV[y] for y in y_test])

    # Evaluate on training set
    y_train_pred = model.predict(x_train)
    y_train_pred = [CLASS_MAP_INV[y] for y in y_train_pred]
    train_f1 = f1_score(y_train, y_train_pred, average="weighted")
    log.debug("Training F1 Score: %.4f", train_f1)
    print_confusion_matrix_grid(y_train, y_train_pred)

    # Evaluate on test set
    y_test_pred = model.predict(x_test)
    y_test_pred = [CLASS_MAP_INV[y] for y in y_test_pred]
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")
    log.debug("Test F1 Score: %.4f", test_f1)

    # Print confusion matrix in clean grid format
    print_confusion_matrix_grid(y_test, y_test_pred)

    # Print training summary
    log.info("\n" + "=" * 50)
    log.info("TRAINING SUMMARY")
    log.info("=" * 50)
    log.info("Model Type: %s", model_type.value)
    log.info("Training Samples: %s", len(x_train))
    log.info("Test Samples: %s", len(x_test))
    log.info("Number of Features: %s", len(available_columns))
    log.info("Number of Classes: %s", len(np.unique(y_train)))
    log.debug("Training F1 Score: %.4f", train_f1)
    log.debug("Test F1 Score: %.4f", test_f1)
    log.info("Training Accuracy: %.4f", model.score(x_train, y_train))
    log.info("Test Accuracy: %.4f", model.score(x_test, y_test))
    log.info("=" * 50)

    return x_train, x_test, y_train, y_test, model, available_columns, class_centroid


def save_model(
    model: Union[RandomForestClassifier, xgboost.XGBClassifier], filepath: str
) -> None:
    """Save the trained model to disk."""
    joblib.dump(model, filepath)
    log.info("Model saved to %s", filepath)


def predict_single(
    model: Union[RandomForestClassifier, xgboost.XGBClassifier],
    features: Union[List[float], pd.Series],
) -> int:
    """Make a prediction on a single sample."""
    return int(model.predict([features])[0])


def predict_proba_single(
    model: Union[RandomForestClassifier, xgboost.XGBClassifier],
    features: Union[List[float], pd.Series],
) -> np.ndarray:
    """Get prediction probabilities for a single sample."""
    proba = model.predict_proba([features])[0]
    return np.array(proba, dtype=np.float64)


def print_confusion_matrix_grid(y_true: pd.Series, y_pred: np.ndarray) -> None:
    """
    Print confusion matrix in a clean, easy-to-read grid format.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names to display instead of numbers
    """
    from sklearn.metrics import confusion_matrix

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Get unique classes
    classes = sorted(np.unique(y_true))
    class_names = [TextClass(i).name for i in classes]
    max_name_len = max(len(name) for name in class_names) + 1

    # Calculate metrics
    total_samples = cm.sum()
    accuracy = np.trace(cm) / total_samples

    log.info("\n" + "=" * 80)
    log.info("CONFUSION MATRIX")
    log.info("=" * 80)
    log.info("Accuracy: %.4f (%s/%s correct)", accuracy, np.trace(cm), total_samples)
    log.info("")

    # Print header
    header = f"{'Predicted ->':^{max_name_len}}"
    for i, name in enumerate(class_names):
        header += f"{name:^{max_name_len}}"
    log.info(header)
    log.info("-" * (max_name_len + max_name_len * len(classes)))

    # Print matrix rows
    for i, true_class in enumerate(classes):
        row = f"{class_names[i]:^{max_name_len}}"
        for j, pred_class in enumerate(classes):
            count = cm[i, j]
            # Highlight correct predictions (diagonal)
            if i == j:
                # Correct predictions
                row += f"{count:^{max_name_len}}"
            else:
                # Incorrect predictions
                row += f"{count:^{max_name_len}}"
        log.info(row)

    log.info("-" * (max_name_len + max_name_len * len(classes)))

    # Print per-class metrics
    log.info("\nPer-Class Metrics:")
    log.info("-" * 75)
    log.info(
        f"{'SourceClass':<{max_name_len}} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'# Samples':<10}"
    )
    log.info("-" * 75)

    for i, class_name in enumerate(class_names):
        # Calculate per-class metrics
        # True positives
        tp = cm[i, i]
        # False positives
        fp = cm[:, i].sum() - tp
        # False negatives
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        num_samples = f"{tp}/{cm[i, :].sum()}"

        log.info(
            f"{class_name:<{max_name_len}} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {num_samples:<10}"
        )

    log.info("=" * 80)


def plot_validation_loss_vs_n_estimators(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators_range: List[int] = None,
    random_state: int = 42,
) -> plt.Figure:
    """
    Plot validation loss vs number of trees for Random Forest models.

    This function trains multiple Random Forest models with different numbers of estimators
    and plots the validation loss to help determine the optimal number of trees.

    Args:
        x_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        n_estimators_range: List of n_estimators values to test. Defaults to [10, 25, 50, 100, 200, 300, 400, 500]
        random_state: Random seed for reproducibility

    Returns:
        plt.Figure: The matplotlib figure object
    """
    if n_estimators_range is None:
        n_estimators_range = [10, 25, 50, 100, 200, 300, 400, 500]

    validation_losses = []
    oob_errors = []

    log.info(
        "Training Random Forest models with different numbers of estimators for validation loss..."
    )
    log.info("=" * 70)

    for n_estimators in n_estimators_range:
        log.debug("Training Random Forest with %s estimators...", n_estimators)

        # Train Random Forest with OOB score enabled
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            oob_score=True,
            n_jobs=-1,  # Use all available cores
            max_samples=0.8,  # Use 80% of samples for each tree to ensure OOB samples
            bootstrap=True,  # Ensure bootstrapping is enabled
        )

        rf.fit(x_train.astype(np.float32), y_train.astype(np.int32))

        # Calculate validation loss (1 - accuracy on test set)
        validation_accuracy = rf.score(x_test, y_test)
        validation_loss = 1 - validation_accuracy
        validation_losses.append(validation_loss)

        log.debug(
            "  Validation Loss: %.4f (Accuracy: %.4f)",
            validation_loss,
            validation_accuracy,
        )

        # Calculate OOB error rate (1 - OOB score)
        oob_error = 1 - rf.oob_score_
        oob_errors.append(oob_error)

        log.debug("  OOB Error Rate: %.4f (OOB Score: %.4f)", oob_error, rf.oob_score)

        # Add diagnostic information if OOB score is suspicious
        if rf.oob_score >= 0.999:
            log.warning(
                "    WARNING: OOB score is very high (%.4f). This may indicate:",
                rf.oob_score,
            )
            log.warning("    - Overfitting to training data")
            log.warning("    - Dataset too small for reliable OOB estimation")
            log.warning("    - Features are too predictive of the target")

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 8), sharex=True)
    ax1.plot(n_estimators_range, oob_errors, "bo-", linewidth=2, markersize=8)
    ax1.set_ylabel("Out-of-Bag Error Rate", fontsize=12)
    ax1.set_title(
        "Random Forest: OOB Error Rate vs Number of Trees",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    # Add annotations for the best performing model
    best_idx = np.argmin(oob_errors)
    best_n_estimators = n_estimators_range[best_idx]
    best_oob_error = oob_errors[best_idx]

    # Add some styling
    plt.tight_layout()

    log.info(
        "\nBest number of trees: %s with OOB error rate: %.4f",
        best_n_estimators,
        best_oob_error,
    )
    log.info("=" * 60)

    # Create the plot
    ax2.plot(n_estimators_range, validation_losses, "ro-", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Trees (n_estimators)", fontsize=12)
    ax2.set_ylabel("Validation Loss (1 - Accuracy)", fontsize=12)
    ax2.set_title(
        "Random Forest: Validation Loss vs Number of Trees",
        fontsize=14,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)

    # Add annotations for the best performing model
    best_idx = np.argmin(validation_losses)
    best_n_estimators = n_estimators_range[best_idx]
    best_validation_loss = validation_losses[best_idx]

    # Add some styling
    plt.tight_layout()

    log.info(
        "\nBest number of trees: %s with validation loss: %.4f",
        best_n_estimators,
        best_validation_loss,
    )
    log.info("=" * 70)

    return fig


def plot_xgboost_convergence(
    model: xgboost.XGBClassifier,
) -> plt.Figure:
    """
    Plot XGBoost model convergence showing training and validation loss over epochs.

    This function trains an XGBoost model and plots the convergence curves to help
    identify overfitting/underfitting and determine optimal early stopping.

    Args:
        x_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        plt.Figure: The matplotlib figure object
    """

    log.info("Training XGBoost model for convergence analysis...")
    log.info("=" * 60)

    results = model.evals_result()
    # Extract training and validation losses
    train_losses = results["validation_0"]["mlogloss"]
    val_losses = results["validation_1"]["mlogloss"]
    epochs = list(range(1, len(train_losses) + 1))

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot both training and validation losses
    ax.plot(epochs, train_losses, "bo-", linewidth=2, label="Training Loss")
    ax.plot(epochs, val_losses, "ro-", linewidth=2, label="Validation Loss")

    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Multi-class Log Loss", fontsize=12)
    ax.set_title(
        "XGBoost: Training vs Validation Loss Convergence",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Add annotations for the best performing epoch (lowest validation loss)
    best_epoch = np.argmin(val_losses) + 1  # +1 because epochs start at 1
    best_val_loss = val_losses[best_epoch - 1]
    best_train_loss = train_losses[best_epoch - 1]

    # Add some styling
    plt.tight_layout()

    log.info("\nBest epoch: %s", best_epoch)
    log.info("  Training Loss: %.4f", best_train_loss)
    log.info("  Validation Loss: %.4f", best_val_loss)
    log.info("  Gap (Overfitting): %.4f", best_train_loss - best_val_loss)
    log.info("=" * 60)

    return fig


def main() -> None:
    log.info("Starting multiclass training...")
    log.info("=" * 50)

    cfr_csv_paths = list(CFR_DIR.glob("labelled_pdf/*.csv"))
    easa_csv_paths = [
        EASA_DIR / "labelled_pdf" / f"{pdf_path}.csv"
        for pdf_path in [
            "Easy Access Rules for ATM_ANS Equipment _Regulations _EU_ 2023_1769 _ _EU_ 2023_1768_ _PDF_",
            "Easy Access Rules for small category VCA _PDF_",
            "Easy Access Rules for Large Aeroplanes _CS 25_ _Amendment 27_ _PDF_",
            "Easy Access Rules for Master Minimum Equipment List _CS_MMEL_ _Issue 3_ _PDF_",
            "Easy Access Rules for Airborne Communications_ Navigation and Surveillance _CS_ACNS_ Issue 4 _pdf_",
            "Easy Access Rules for Large Rotorcraft _CS_29_ _Amendment 11_ _PDF_",
            "Easy Access Rules for Normal_Category Aeroplanes _CS_23_ _CS Amendment 6_ AMC_GM Issue 4_ _PDF_",
            "Easy Access Rules for Small Rotorcraft _CS_27_ Amendment 10 _pdf_",
            "Easy Access Rules for U_space _PDF_",
            "Easy Access Rules for Aerodromes _PDF_",
            "Easy Access Rules for Information Security _PDF_",
            "Easy Access Rules for Aircrew _Regulation _EU_ No 1178_2011_ _PDF_",
        ]
    ]
    latex_dir = DATA_DIR / "documents" / "Latex" / "labelled_pdf"
    latex_csv_paths = list(latex_dir.glob("*.csv"))
    latex_bullets_dir = DATA_DIR / "documents" / "Bullets" / "labelled_pdf"
    latex_bullets_csv_paths = list(latex_bullets_dir.glob("*.csv"))

    aus_mdr_dir = DATA_DIR / "documents" / "AUSMDR" / "labelled_pdf"
    aus_mdr_csv_paths = list(aus_mdr_dir.glob("*.csv"))

    def read_dfs(csv_paths: List[Path]) -> Tuple[pd.DataFrame, int]:
        all_dfs = []
        total_pages = 0
        for csv_path in csv_paths:
            parent_dir = csv_path.parent
            log.debug(f"Reading {csv_path}")
            df = pd.read_csv(csv_path)
            df["parent_dir"] = parent_dir
            df["pdf_path"] = csv_path.stem
            num_pages = len(np.unique(df["PageNumber"]))
            log.debug("Num Pages: %s - %s", num_pages, csv_path.stem)
            all_dfs.append(df)
            total_pages += num_pages
        return all_dfs, total_pages

    cfr_dfs, cfr_total_pages = read_dfs(cfr_csv_paths)
    easa_dfs, easa_total_pages = read_dfs(easa_csv_paths)
    latex_dfs, latex_total_pages = read_dfs(latex_csv_paths)
    latex_bullets_dfs, latex_bullets_total_pages = read_dfs(latex_bullets_csv_paths)
    aus_mdr_dfs, aus_mdr_total_pages = read_dfs(aus_mdr_csv_paths)
    # duplicate latex_dfs 2 times
    latex_dfs = latex_dfs * 2
    latex_bullets_dfs = latex_bullets_dfs * 2
    latex_total_pages = latex_total_pages * 2
    latex_bullets_total_pages = latex_bullets_total_pages * 2

    log.info("CFR Total Pages: %s", cfr_total_pages)
    log.info("EASA Total Pages: %s", easa_total_pages)
    log.info("Latex Total Pages: %s", latex_total_pages)
    log.info("Latex Bullets Total Pages: %s", latex_bullets_total_pages)
    log.info("AUS MDR Total Pages: %s", aus_mdr_total_pages)

    df = pd.concat(cfr_dfs + easa_dfs + latex_dfs + latex_bullets_dfs + aus_mdr_dfs)

    if TRAIN_MODEL:
        # Train the model
        (
            x_train,
            x_test,
            y_train,
            y_test,
            model,
            feature_columns,
            class_centroid,
        ) = train_multiclass(
            df=df,
            test_size=0.2,
            random_state=42,
            model_type=model_type,
        )

        log.info("\n" + "=" * 50)
        log.info("Training completed!")
        log.info("Model trained on %s samples", len(x_train))
        log.info("Tested on %s samples", len(x_test))
        log.info("Used %s features", len(feature_columns))
        # Save the model
        model_dir = data_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        if model_type == ModelType.RANDOM_FOREST:
            model_path = model_dir / "RandomForestClassifier.sav"
        elif model_type == ModelType.XGBOOST:
            model_path = model_dir / "XGBoostClassifier.sav"
        save_model(model, model_path)
        class_centroid.to_csv(model_dir / "class_centroid.csv", index=False)

        # Make a prediction on a sample from test set
        if len(x_test) > 0:
            sample_features = x_test.iloc[0:1]
            prediction = model.predict(sample_features)
            log.debug("Sample prediction: %s", prediction[0])
            # log.debug("Actual label: %s", y_test.iloc[0])

        log.info("\nFeature importance (top 10):")
        feature_importance = list(zip(feature_columns, model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

    for feature, importance in feature_importance[:10]:
        log.info("Feature Importance: %12s: %.4f", feature, importance)

    # print OOB and MSE for Random Forest
    log.info("OOB Error: %.4f", model.oob_score_)
    log.info("MSE: %.4f", model.score(x_train, y_train))
    exit()
    # Create plots directory
    plots_dir = data_dir / "output_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    # Plot OOB error vs number of trees for Random Forest models
    if model_type == ModelType.RANDOM_FOREST:
        log.info("\n" + "=" * 50)
        log.info("PLOTTING OOB ERROR VS NUMBER OF TREES")
        log.info("=" * 50)

        # Plot validation loss vs number of trees
        log.info("\n" + "=" * 50)
        log.info("PLOTTING VALIDATION LOSS VS NUMBER OF TREES")
        log.info("=" * 50)

        val_fig = plot_validation_loss_vs_n_estimators(
            x_train,
            y_train,
            x_test,
            y_test,
            n_estimators_range=[
                10,
                25,
                50,
                100,
                200,
            ],  # Smaller range for faster execution
            random_state=42,
        )

        # Save validation loss plot
        val_plot_path = plots_dir / "random_forest_validation_loss_vs_n_estimators.png"
        val_fig.savefig(val_plot_path, dpi=300, bbox_inches="tight")
        log.info("Validation loss plot saved to: %s", val_plot_path)
        plt.close(val_fig)

    elif model_type == ModelType.XGBOOST:
        # Plot XGBoost convergence
        log.info("\n" + "=" * 50)
        log.info("PLOTTING XGBOOST CONVERGENCE")
        log.info("=" * 50)

        convergence_fig = plot_xgboost_convergence(model)

        # Save XGBoost convergence plot
        convergence_plot_path = plots_dir / "xgboost_convergence.png"
        convergence_fig.savefig(convergence_plot_path, dpi=300, bbox_inches="tight")
        log.info("XGBoost convergence plot saved to: %s", convergence_plot_path)
        plt.close(convergence_fig)


# Example driver code
if __name__ == "__main__":
    from enginius_parser import CFR_DIR, DATA_DIR, EASA_DIR

    logging.basicConfig(level=logging.DEBUG)

    data_dir = DATA_DIR / "documents"
    TRAIN_MODEL = True
    model_type = ModelType.RANDOM_FOREST
    main()
