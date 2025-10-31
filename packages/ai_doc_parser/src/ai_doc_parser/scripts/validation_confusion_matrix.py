"""
Script to generate validation confusion matrix using real data from validation directories.

This script loads labelled and inference data from the validation directories and
generates a confusion matrix to evaluate model performance.
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from ai_doc_parser.text_class import TextClass
from ai_doc_parser.training.classifier_trainer import print_confusion_matrix_grid

log = logging.getLogger(__name__)


def load_validation_data() -> tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Load labelled and inference data from validation directories.

    Returns:
        tuple: (labelled_dfs, inference_dfs) lists of dataframes
    """
    # Define paths
    labelled_dir = Path("C:/Users/r123m/Documents/enginius/source/ai-pdf-parser/data/documents/validation/labelled_pdf")
    inference_dir = Path(
        "C:/Users/r123m/Documents/enginius/source/ai-pdf-parser/data/documents/validation/ai_parsed_pdf_not_combined"
    )

    labelled_dfs = []
    inference_dfs = []

    # Check if directories exist
    if not labelled_dir.exists():
        print(f"Warning: Labelled directory not found: {labelled_dir}")
        return labelled_dfs, inference_dfs

    if not inference_dir.exists():
        print(f"Warning: Inference directory not found: {inference_dir}")
        return labelled_dfs, inference_dfs

    # Get all CSV files from both directories
    labelled_files = list(labelled_dir.glob("*.csv"))
    inference_files = list(inference_dir.glob("*.csv"))

    print(f"Found {len(labelled_files)} labelled files and {len(inference_files)} inference files")

    # Load labelled data
    for file_path in labelled_files:
        # if file_path.name == "devices-argument.csv":
        #     continue
        try:
            df = pd.read_csv(file_path)
            if 'LabelledClass' in df.columns and 'pdf_idx' in df.columns:
                # Add the CSV file path as a column
                df['csv_path'] = str(file_path)
                labelled_dfs.append(df)
                print(f"Loaded labelled data: {file_path.name} ({len(df)} rows)")
            else:
                print(f"Warning: {file_path.name} missing required columns (LabelledClass, pdf_idx)")
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")

    # Load inference data
    for file_path in inference_files:
        try:
            df = pd.read_csv(file_path)
            if 'FinalClass' in df.columns and 'pdf_idx' in df.columns:
                # Add the CSV file path as a column
                df['csv_path'] = str(file_path)
                inference_dfs.append(df)
                print(f"Loaded inference data: {file_path.name} ({len(df)} rows)")
            else:
                print(f"Warning: {file_path.name} missing required columns (FinalClass, pdf_idx)")
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")

    # If no files found in validation directories, try to find matching files
    if not labelled_dfs or not inference_dfs:
        print("No data found in validation directories. Checking for alternative data sources...")

        # Try to find data in other directories
        data_root = Path("C:/Users/r123m/Documents/enginius/source/ai-pdf-parser/data/documents")

        # Look for CFR data as an example
        cfr_labelled_dir = data_root / "CFR" / "labelled_pdf"
        cfr_inference_dir = data_root / "CFR" / "ai_parsed_pdf"

        if cfr_labelled_dir.exists() and cfr_inference_dir.exists():
            print("Using CFR data as example...")

            # Load first few CFR files as example
            cfr_labelled_files = list(cfr_labelled_dir.glob("*.csv"))[:2]  # Limit to 2 files
            cfr_inference_files = list(cfr_inference_dir.glob("*.csv"))[:2]

            for file_path in cfr_labelled_files:
                try:
                    df = pd.read_csv(file_path)
                    if 'LabelledClass' in df.columns and 'pdf_idx' in df.columns:
                        # Add the CSV file path as a column
                        df['csv_path'] = str(file_path)
                        labelled_dfs.append(df)
                        print(f"Loaded CFR labelled data: {file_path.name} ({len(df)} rows)")
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")

            for file_path in cfr_inference_files:
                try:
                    df = pd.read_csv(file_path)
                    if 'FinalClass' in df.columns and 'pdf_idx' in df.columns:
                        # Add the CSV file path as a column
                        df['csv_path'] = str(file_path)
                        inference_dfs.append(df)
                        print(f"Loaded CFR inference data: {file_path.name} ({len(df)} rows)")
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")

    return labelled_dfs, inference_dfs


def remove_list_cont(input_class: TextClass) -> TextClass:
    if input_class in [TextClass.ENUM_LIST, TextClass.BULLET_LIST]:
        return TextClass.PARAGRAPH
    if input_class in [TextClass.ENUM_LIST_CONT, TextClass.BULLET_LIST_CONT, TextClass.GEN_LIST_CONT]:
        return TextClass.PARAGRAPH_CONT
    return input_class


def print_validation_confusion_matrix(
    labelled_dfs: List[pd.DataFrame], inference_dfs: List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Print confusion matrix for validation data by comparing labelled and inference dataframes.

    Args:
        labelled_dfs: List of dataframes containing labelled data with 'LabelledClass' column
        inference_dfs: List of dataframes containing inference results with 'FinalClass' column

    Returns:
        pd.DataFrame: DataFrame containing all mismatches with columns:
                     - csv_path: Path to the input CSV file
                     - pdf_idx: Index of the row in the PDF
                     - text: Text content
                     - labelled_class: True class from labelled data
                     - inference_class: Predicted class from inference data
    """
    if len(labelled_dfs) != len(inference_dfs):
        raise ValueError("labelled_dfs and inference_dfs must have the same length")

    if len(labelled_dfs) == 0:
        print("Warning: No dataframes provided for validation")
        return pd.DataFrame()

    print(f"Computing validation confusion matrix across {len(labelled_dfs)} document pairs")

    # Collect all true and predicted labels
    all_true_labels = []
    all_predicted_labels = []

    # Collect mismatch data
    mismatch_data = []

    for i, (labelled_df, inference_df) in enumerate(zip(labelled_dfs, inference_dfs)):
        print(f"Processing document pair {i}: {len(labelled_df)} labelled rows, {len(inference_df)} inference rows")

        # Create a mapping from pdf_idx to FinalClass for quick lookup
        inference_lookup = dict(zip(inference_df['pdf_idx'], inference_df['FinalClass']))

        # Get the CSV path for this document pair from the stored csv_path column
        csv_path = f"document_{i}"  # Default fallback
        if 'csv_path' in labelled_df.columns and len(labelled_df) > 0:
            csv_path = labelled_df['csv_path'].iloc[0]

        # For each row in labelled_df, find the corresponding inference result
        for _, row in labelled_df.iterrows():
            pdf_idx = row['pdf_idx']
            labelled_class = row['LabelledClass']
            text = row.get('text', '')  # Get text if available

            labelled_class = remove_list_cont(labelled_class)

            # Skip if labelled class is NaN
            if pd.isna(labelled_class):
                continue

            # Find the corresponding inference result
            if pdf_idx in inference_lookup:
                predicted_class = inference_lookup[pdf_idx]

                predicted_class = remove_list_cont(predicted_class)

                # Skip if predicted class is NaN
                if pd.isna(predicted_class):
                    continue

                all_true_labels.append(labelled_class)
                all_predicted_labels.append(predicted_class)

                # Check if this is a mismatch
                if labelled_class != predicted_class:
                    mismatch_data.append(
                        {
                            'csv_path': csv_path,
                            'pdf_idx': pdf_idx,
                            'text': text,
                            'labelled_class': labelled_class,
                            'inference_class': predicted_class,
                        }
                    )
            else:
                print(f"No inference result found for pdf_idx {pdf_idx} in document {i}")

    if len(all_true_labels) == 0:
        print("Warning: No matching labels found between labelled and inference data")
        return pd.DataFrame()

    print(f"Found {len(all_true_labels)} matching label pairs for validation")
    print(f"Found {len(mismatch_data)} mismatches")

    # Convert to pandas Series for compatibility with existing function
    y_true = pd.Series(all_true_labels)
    y_pred = np.array(all_predicted_labels)

    # Use the imported confusion matrix function from classifier_trainer
    print_confusion_matrix_grid(y_true, y_pred)

    # Return mismatch DataFrame
    return pd.DataFrame(mismatch_data)


def main():
    """
    Load validation data and generate confusion matrix.
    """
    print("Loading validation data...")
    labelled_dfs, inference_dfs = load_validation_data()

    if not labelled_dfs or not inference_dfs:
        print("No data found to generate confusion matrix.")
        return

    if len(labelled_dfs) != len(inference_dfs):
        print(f"Warning: Mismatch in number of files - {len(labelled_dfs)} labelled vs {len(inference_dfs)} inference")
        # Use the minimum number of files
        min_files = min(len(labelled_dfs), len(inference_dfs))
        labelled_dfs = labelled_dfs[:min_files]
        inference_dfs = inference_dfs[:min_files]
        print(f"Using {min_files} file pairs for comparison")

    print(f"\nGenerating confusion matrix for {len(labelled_dfs)} document pairs...")

    # Show sample data
    if labelled_dfs:
        print("\nSample labelled data:")
        print(labelled_dfs[0][['pdf_idx', 'text', 'LabelledClass']].head())

    if inference_dfs:
        print("\nSample inference data:")
        print(inference_dfs[0][['pdf_idx', 'text', 'FinalClass']].head())

    print("\n" + "=" * 80)
    print("VALIDATION CONFUSION MATRIX")
    print("=" * 80)

    # Generate confusion matrix and get mismatch data
    mismatch_df = print_validation_confusion_matrix(labelled_dfs, inference_dfs)

    for i, (file_name, file_path_df) in enumerate(mismatch_df.groupby('csv_path')):
        num_total = len(labelled_dfs[i])
        print(f"{len(file_path_df)}/{num_total} ({len(file_path_df)/num_total*100:.2f}%) Mismatches in {file_name}")

    # mismatch_df = mismatch_df[mismatch_df['labelled_class'] == TextClass.PARAGRAPH]
    # mismatch_df = mismatch_df[mismatch_df['inference_class'] == TextClass.PARAGRAPH_CONT]
    mismatch_df = mismatch_df[
        mismatch_df['csv_path']
        == r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\validation\labelled_pdf\CFR-2023-title14-vol5.csv"
    ]

    mismatch_dir = Path(r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\validation\mismatch")
    mismatch_df.to_csv(mismatch_dir / "CFR-2023-title14-vol5.csv", index=False)
    #

    # Display mismatch information
    if not mismatch_df.empty:
        print(f"\nFound {len(mismatch_df)} mismatches:")
        print("=" * 80)
        print("MISMATCH DETAILS")
        print("=" * 80)

        # Show first 10 mismatches as examples
        sample_mismatches = mismatch_df.head(1)
        for idx, row in sample_mismatches.iterrows():
            print(f"CSV: {row['csv_path']}")
            print(f"PDF Index: {row['pdf_idx']}")
            print(f"Text: {row['text'][:100]}{'...' if len(str(row['text'])) > 100 else ''}")
            print(f"Labelled Class: {TextClass(row['labelled_class']).name}")
            print(f"Inference Class: {TextClass(row['inference_class']).name}")
            print("-" * 40)

        if len(mismatch_df) > 10:
            print(f"... and {len(mismatch_df) - 10} more mismatches")

        # Save mismatches to CSV
        output_path = Path("validation_mismatches.csv")
        mismatch_df.to_csv(output_path, index=False)
        print(f"\nAll mismatches saved to: {output_path}")
    else:
        print("\nNo mismatches found - perfect classification!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
