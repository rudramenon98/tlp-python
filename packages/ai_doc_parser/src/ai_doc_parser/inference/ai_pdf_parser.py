from typing import Tuple
import csv
import json
import logging
import os
import re
import sys
import time
import traceback
from datetime import date, datetime
from multiprocessing import current_process
from multiprocessing.pool import ThreadPool as Pool
from pathlib import Path
from typing import Any, Dict, List

import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from ai_doc_parser.inference.feature_computation.feature_computer import compute_features
from ai_doc_parser.inference.pdf_extraction.pymu_extractor import extract_pdf_text
from ai_doc_parser.inference.post_classification_heuristics import post_classification_heuristics
from ai_doc_parser.text_class import AI_PARSED_CLASSES, CLASS_MAP_INV, CONTINUE_PAIRS, TextClass
from ai_doc_parser.tools.model_interpretability import ModelInterpretabilityAnalyzer
from ai_doc_parser.training.classifier_trainer import prepare_df_for_model

log = logging.getLogger(__name__)
loaded_model = None


def replace_gen_list_cont(pdf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort df by pdf_idx and replace gen_list_cont with the appropriate class based on the previous blocks.
    """
    pdf_df = pdf_df.sort_values(by='pdf_idx')
    cont_type = TextClass.BULLET_LIST_CONT
    for i, row in pdf_df.iterrows():
        if row['PredictedClass'] == TextClass.BULLET_LIST:
            cont_type = TextClass.BULLET_LIST_CONT
        elif row['PredictedClass'] == TextClass.ENUM_LIST:
            cont_type = TextClass.ENUM_LIST_CONT
        elif row['PredictedClass'] == TextClass.GEN_LIST_CONT:
            pdf_df.loc[i, 'FinalClass'] = cont_type
            pdf_df.loc[i, 'FinalClassName'] = cont_type.name
    return pdf_df


def combine_rows(row1: dict, row2: dict) -> dict:
    """
    Combine two rows by merging their text and updating bounding box coordinates.

    Args:
        row1: First row (base row) - typically contains the base class
        row2: Second row (continuation row) - typically contains the continuation class

    Returns:
        dict: Combined row with merged text and updated coordinates
    """
    # Start with a copy of the first row as the base
    res_dict = row1.copy()

    # Concatenate the text from both rows
    res_dict['text'] = row1['text'] + row2['text']

    # Keep the predicted class and class name from the first row (base class)
    res_dict['FinalClass'] = row1['FinalClass']

    # For minimum coordinate columns, take the minimum value to get the top-left boundary
    min_cols = ['x0', 'y0', 'block_x0', 'block_y0', 'line_x0', 'line_y0']
    for col in min_cols:
        res_dict[col] = min(row1[col], row2[col])

    # For maximum coordinate columns, take the maximum value to get the bottom-right boundary
    max_cols = ['x1', 'y1', 'block_x1', 'block_y1', 'line_x1', 'line_y1']
    for col in max_cols:
        res_dict[col] = max(row1[col], row2[col])

    return res_dict


def combine_text_block(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine text blocks that are continuations of each other based on CONTINUE_PAIRS.

    This method goes through the dataframe and finds pairs where one row represents
    a line that is continued from the previous line. It combines these pairs into
    single rows and returns a new dataframe.

    Args:
        df: DataFrame with text blocks and their predicted classes

    Returns:
        pd.DataFrame: New DataFrame with combined text blocks
    """
    # Initialize list to store the final result rows
    result_rows: List[dict] = []
    # Track the current row being built (for continuation pairs)
    new_row: Dict | None = None
    # Current index in the dataframe
    idx = 0

    cont_class_map = {cont_class: first_line_class for first_line_class, cont_class in CONTINUE_PAIRS}
    last_page_no = -1
    last_col_num = -1

    # Iterate through all rows in the dataframe
    while idx < len(df):
        # Get current row
        row = df.iloc[idx]

        if last_page_no != row['PageNumber'] or last_col_num != row['col_num']:
            if new_row is not None:
                result_rows.append(new_row)
                new_row = None
            if row['FinalClass'] in cont_class_map.keys():
                df.loc[df.index[idx], 'FinalClass'] = cont_class_map[row['FinalClass']]
        row = df.iloc[idx]
        last_page_no = row['PageNumber']
        last_col_num = row['col_num']

        final_class = row['FinalClass']

        # Case 1: Special classes that should always be added as-is (no continuation)
        if final_class in [TextClass.HEADER, TextClass.FOOTER, TextClass.FOOT_NOTE]:
            # Dont reset the new_row
            result_rows.append(row)

        # Case 2: Found a base class that can have continuations
        elif final_class in cont_class_map.values():
            # If we already have a pending new_row, add it first before starting a new one
            if new_row is not None:
                result_rows.append(new_row)
            # Start building a new row that might have continuations
            new_row = row.copy()

        # Case 3: Found a continuation class and we have a base row to continue
        elif final_class in cont_class_map.keys() and new_row is not None:
            # Combine the continuation row with the base row
            new_row = combine_rows(new_row, row)

        # Case 4: Any other class or no continuation match
        else:
            # If we have a pending new_row, add it first
            if new_row is not None:
                result_rows.append(new_row)
                new_row = None
            # Add the current row
            if row['FinalClass'] not in cont_class_map.keys():
                result_rows.append(row)

        # Move to next row
        idx += 1

    # After the loop, check if there's a pending new_row that hasn't been added
    if new_row is not None:
        result_rows.append(new_row)

    # Convert the list of rows back to a DataFrame
    result_df = pd.DataFrame(result_rows)
    return result_df


def classify_pdf_ai(
    pdf_df: pd.DataFrame, model: RandomForestClassifier | XGBClassifier
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Classify a PDF document using the trained model.

    Args:
        document_path: Path to the PDF document
        model_path: Path to the trained model file

    Returns:
        pd.DataFrame: DataFrame with predictions for each text block
    """
    # Load the trained model
    # pdf_df = extract_pdf_text(document_path, "EASA")
    # pdf_df = compute_features(pdf_df)
    # pdf_df = pd.read_csv(document_path.parent / "computed_features" / f"{document_path.stem}.csv")

    pdf_df['original_index'] = pdf_df.index

    # drop rows were text is empty or nan
    pdf_df = pdf_df[pdf_df['text'].notna()]
    pdf_df = pdf_df[pdf_df['text'] != '']

    # Reset index to ensure sequential indexing after filtering
    pdf_df = pdf_df.reset_index(drop=True)

    # drop rows that have an ExtractedClass that is not nan, they will be added back later
    non_ai_class_df = pdf_df[~pdf_df['ExtractedClass'].isin(AI_PARSED_CLASSES)]
    non_ai_class_df = non_ai_class_df[~np.isnan(non_ai_class_df['ExtractedClass'])]

    pdf_df = pdf_df[pdf_df['ExtractedClass'].isna()]
    # Use our prepare_df_for_model function
    prepared_df, feature_columns = prepare_df_for_model(df=pdf_df, add_shifted_features=True, verbose=True)
    prepared_features = prepared_df[feature_columns]

    # Make predictions using the model
    print(prepared_features.head())
    predictions = model.predict(prepared_features)
    probabilities = model.predict_proba(prepared_features)

    # Convert predictions to class names
    predictions = [CLASS_MAP_INV[pred] for pred in predictions]

    # combine predictions to original dataframe

    # Add predictions back to the original dataframe
    pdf_df['PredictedClass'] = predictions

    # Apply heuristics (deprecated in this module)
    probability_strs = ['[' + ", ".join(list(map(lambda p: str(round(p, 3)), prob))) + ']' for prob in probabilities]
    pdf_df['PredictionProbs'] = probability_strs
    pdf_df['MaxPredictionProb'] = probabilities.max(axis=1)

    # Add back the predicted class name
    pdf_df['PredictedClassName'] = pdf_df['PredictedClass'].map(lambda x: TextClass(x).name)

    ai_results_no_heuristics = pdf_df.copy()

    # Add back the rows that have an ExtractedClass that is not nan
    pdf_df = pd.concat([pdf_df, pdf_df[pdf_df['ExtractedClass'].notna()]])

    # Make FinalClass the ExtractedClass if it is not nan else the PredictedClass
    pdf_df['FinalClass'] = pdf_df['ExtractedClass'].fillna(pdf_df['PredictedClass'])

    # sort by pdf_idx
    pdf_df = pdf_df.sort_values(by='pdf_idx')

    # # replace gen_list_cont with the appropriate class based on the previous blocks
    pdf_df = replace_gen_list_cont(pdf_df)

    # apply post classification heuristics
    pdf_df = post_classification_heuristics(pdf_df)
    
    results_not_combined = pdf_df.copy()

    # combine text blocks
    pdf_df = combine_text_block(pdf_df)

    # apply post classification heuristics
    pdf_df = post_classification_heuristics(pdf_df)

    pdf_df['FinalClassName'] = pdf_df['FinalClass'].map(lambda x: TextClass(x).name)

    # Add back the extracted classes that do not require ML classification
    non_ai_class_df['FinalClass'] = non_ai_class_df['ExtractedClass']
    non_ai_class_df['FinalClassName'] = non_ai_class_df['ExtractedClass'].map(lambda x: TextClass(x).name)
    pdf_df = pd.concat([pdf_df, non_ai_class_df])
    pdf_df = pdf_df.sort_values(by='original_index')
    pdf_df = pdf_df.drop(columns=['original_index'])

    return prepared_features, pdf_df, ai_results_no_heuristics, results_not_combined


def load_model(model_path: str | Path) -> RandomForestClassifier:
    """Load a trained model from disk."""
    return joblib.load(model_path)


def parse_pdf_ai(
    document_path: Path,
    model: RandomForestClassifier | XGBClassifier,
    pdf_type: str = "",
    computed_features_df: pd.DataFrame | None = None,
    extracted_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Parse a PDF document using the trained model.
    """
    if isinstance(document_path, str):
        document_path = Path(document_path)
    if computed_features_df is None:
        if extracted_df is None:
            pdf_df = extract_pdf_text(document_path, pdf_type)
        else:
            pdf_df = extracted_df
        pdf_df = compute_features(pdf_df)
    else:
        pdf_df = computed_features_df
    prepared_features, pdf_df, ai_results_no_heuristics, ai_results_not_combined = classify_pdf_ai(pdf_df, model)
    return prepared_features, pdf_df, ai_results_no_heuristics, ai_results_not_combined


def main() -> None:
    from ai_doc_parser import DATA_DIR, EASA_DIR, EASA_PDF, LATEX_PDF

    data_dir = DATA_DIR / "documents"
    overwrite = True

    pdf_path = data_dir / "NIST" / "SP 1299, NIST Cybersecurity Framework 2.0%3A Resourc.pdf"
    # model_path = data_dir / "models" / "RandomForestClassifier.sav"
    model_path = data_dir / "models" / "RandomForestClassifier.sav"
    # model_path = data_dir / "models" / "XGBoostClassifier.sav"
    model = load_model(model_path)
    # parse_pdf_ai(pdf_path, model, "AC")
    final_output_path = pdf_path.parent / "ai_parsed_pdf" / f"{pdf_path.stem}.csv"
    final_output_path.parent.mkdir(parents=True, exist_ok=True)

    extracted_output_path = pdf_path.parent / "pdf_extracted" / f"{pdf_path.stem}.csv"
    extracted_output_path.parent.mkdir(parents=True, exist_ok=True)

    computed_features_output_path = pdf_path.parent / "computed_features" / f"{pdf_path.stem}.csv"
    computed_features_output_path.parent.mkdir(parents=True, exist_ok=True)

    no_heuristics_output_path = pdf_path.parent / "ai_parsed_pdf_no_heuristics" / f"{pdf_path.stem}.csv"
    no_heuristics_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not extracted_output_path.exists() or overwrite:
        pdf_df = extract_pdf_text(pdf_path, "AC")
        pdf_df.to_csv(extracted_output_path, index=False)
    else:
        pdf_df = pd.read_csv(extracted_output_path)

    if not computed_features_output_path.exists() or overwrite:
        pdf_df = compute_features(pdf_df)
        pdf_df.to_csv(computed_features_output_path, index=False)
    else:
        pdf_df = pd.read_csv(computed_features_output_path)

    prepared_features, pdf_df, ai_results_no_heuristics, ai_results_not_combined = classify_pdf_ai(pdf_df, model)

    prepared_features_path = pdf_path.parent / "prepared_features" / f"{pdf_path.stem}.csv"
    prepared_features_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_features.to_csv(prepared_features_path, index=False)
    print(f"Prepared features saved to {prepared_features_path}")

    pdf_df.to_csv(final_output_path, index=False)
    ai_results_no_heuristics.to_csv(no_heuristics_output_path, index=False)
    print(f"AI results no heuristics saved to {no_heuristics_output_path}")

    print(pdf_df)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
# %%
