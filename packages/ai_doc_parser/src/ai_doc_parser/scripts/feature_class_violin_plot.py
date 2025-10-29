from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_feature_violin_plots(csv_path: Path) -> List[plt.Figure]:
    """
    Create violin plots for each class showing the distribution of features.

    Args:
        csv_path (Path): Path to the CSV file containing labelled data

    Returns:
        List[plt.Figure]: List of matplotlib figure objects for all created plots
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Check if required columns exist
    if 'ClassLabel' not in df.columns:
        raise ValueError("CSV must contain a 'ClassLabel' column")

    # Define the feature columns (matching the ones from classifier_trainer.py)
    feature_columns = [
        "first_character_is_bullet",
        "first_word_is_compound",
        "number_dots",
        "fraction_capitalized",
        "last_word_is_roman",
        "ends_with_upper",
        "no_count_consecutive_spaces",
        "consecutive_dots",
        "left_indent_bin",
        "right_space_bin",
        "next_line_space_bin",
        "font_size_bin",
        "no_of_words_bin",
        'text_length_bin',
    ]

    # Filter to only include features that exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]
    missing_features = [col for col in feature_columns if col not in df.columns]

    if missing_features:
        print(f"Warning: The following features are missing from the CSV: {missing_features}")

    if not available_features:
        raise ValueError("No feature columns found in the CSV file")

    print(f"Creating violin plots for {len(available_features)} features across {df['ClassLabel'].nunique()} classes")

    # Get unique classes
    classes = sorted(df['ClassLabel'].unique())

    # Set up the plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)

    # List to store all figures
    figures: List[plt.Figure] = []

    # Create violin plots for each class
    for class_label in classes:
        # Filter data for this class
        class_data = df[df['ClassLabel'] == class_label].copy()

        if len(class_data) == 0:
            continue

        # Calculate the number of rows and columns for subplots
        n_features = len(available_features)

        # Create subplots
        fig, axes = plt.subplots(n_features, 1, figsize=(20, 5 * n_features))
        fig.suptitle(
            f'Feature Distributions for Class: {class_label}\n(n={len(class_data)} samples)',
            fontsize=16,
            fontweight='bold',
            y=0.98,
        )

        # Create histogram for each feature
        for i, feature in enumerate(available_features):
            ax = axes[i]
            feature_data = class_data[feature].dropna()

            if len(feature_data) > 0:
                ax.hist(feature_data, bins=20, alpha=0.7, edgecolor='black', linewidth=0.5)
                ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
                ax.set_xlabel('Value', fontsize=8)
                ax.set_ylabel('Frequency', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', labelsize=8)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{feature}', fontsize=10, fontweight='bold')

        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        # Add some padding
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make room for the suptitle

        # Add figure to the list
        figures.append(fig)

        plt.close()

    # Create a combined plot showing all classes together (for comparison)
    print("Creating combined violin plot for all classes...")

    # Prepare data for combined plot
    combined_data = []
    for class_label in classes:
        class_data = df[df['ClassLabel'] == class_label][available_features].copy()
        class_data_melted = class_data.melt(var_name='Feature', value_name='Value')
        class_data_melted['SourceClass'] = class_label
        combined_data.append(class_data_melted)

    combined_df = pd.concat(combined_data, ignore_index=True)

    # Create combined histogram plot
    # Use a subset of features for the combined plot to avoid overcrowding
    sample_features = available_features[:8] if len(available_features) > 8 else available_features

    n_features = len(sample_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    combined_fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    combined_fig.suptitle('Feature Distributions Across All Classes', fontsize=16, fontweight='bold', y=0.98)

    # Flatten axes array for easier indexing
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    # Create histogram for each feature showing all classes
    for i, feature in enumerate(sample_features):
        ax = axes[i]

        # Get data for this feature across all classes
        for class_label in classes:
            class_data = df[df['ClassLabel'] == class_label][feature].dropna()
            if len(class_data) > 0:
                ax.hist(class_data, bins=20, alpha=0.6, label=class_label, edgecolor='black', linewidth=0.5)

        ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the suptitle

    # Add combined figure to the list
    figures.append(combined_fig)

    plt.close()

    return figures


def main() -> None:
    """Main function to run the violin plot generation."""
    from ai_doc_parser import EASA_PDF

    df_path = EASA_PDF.parent / "labelled_pdf" / f"{EASA_PDF.stem}.csv"

    output_dir = EASA_PDF.parent / "feature_class_violin_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # empty the output directory
    for file in output_dir.glob("*.png"):
        file.unlink()

    figures = create_feature_violin_plots(df_path)

    # Save all figures to the output directory
    for i, fig in enumerate(figures):
        if i < len(figures) - 1:  # Individual class plots
            # Get class label from the title (assuming it's the first part before newline)
            title = fig.axes[0].get_title()
            class_label = title.split('\n')[0].replace('Feature Distributions for Class: ', '')
            filename = f"violin_plot_class_{class_label.replace(' ', '_').replace('/', '_')}.png"
        else:  # Combined plot
            filename = "violin_plot_all_classes.png"

        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_dir / filename}")

    print(f"Violin plot generation completed successfully! Created {len(figures)} plots.")


if __name__ == "__main__":
    main()
