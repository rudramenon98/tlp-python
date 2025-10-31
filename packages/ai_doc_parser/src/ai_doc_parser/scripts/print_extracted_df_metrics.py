"""
Script to analyze and compare extracted labels with PDF labels.

This script reads CSV files containing extracted labels and PDF labels,
then generates a comprehensive comparison report with metrics by text class.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from ai_doc_parser.text_class import TextClass
from ai_doc_parser.training.common_tools import clean_text

COLUMN_WIDTH = 20


def load_dataframes(
    extracted_labels_path: Path, labelled_pdf_path: Path, classified_pdf_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the extracted labels and PDF labels dataframes.

    Returns:
        Tuple containing (extracted_df, labelled_pdf_df)

    Raises:
        FileNotFoundError: If the required CSV files are not found
    """

    if extracted_labels_path.exists():
        extracted_df = pd.read_csv(extracted_labels_path)
    else:
        extracted_df = pd.DataFrame(columns=["text", "SourceClass", "PageNumber"])
        # raise FileNotFoundError(f"Extracted labels file not found: {extracted_labels_path}")

    if labelled_pdf_path.exists():
        labelled_pdf_df = pd.read_csv(labelled_pdf_path)
    else:
        labelled_pdf_df = pd.DataFrame(columns=["text", "LabelledClass", "PageNumber"])
        # raise FileNotFoundError(f"Labelled PDF file not found: {labelled_pdf_path}")

    # remove rows where text is empty or nan
    extracted_df = extracted_df[
        extracted_df["text"].notna() & (extracted_df["text"] != "")
    ]
    labelled_pdf_df = labelled_pdf_df[
        labelled_pdf_df["text"].notna() & (labelled_pdf_df["text"] != "")
    ]

    if not classified_pdf_path.exists():
        classified_pdf_df = pd.DataFrame(
            columns=[
                "text",
                "PredictedClass",
            ]
        )
    else:
        classified_pdf_df = pd.read_csv(classified_pdf_path)
        classified_pdf_df = classified_pdf_df[
            classified_pdf_df["text"].notna() & (classified_pdf_df["text"] != "")
        ]

        # for old parser dfs.
        if "PredictedClass" not in classified_pdf_df.columns:
            classified_pdf_df["PredictedClass"] = classified_pdf_df["pred"]

    return extracted_df, labelled_pdf_df, classified_pdf_df


def get_pdf_matches_to_xml_idx(labelled_pdf_df: pd.DataFrame, xml_idx: int) -> None:
    print(
        labelled_pdf_df[labelled_pdf_df["xml_idx"] == xml_idx][
            ["text", "pdf_idx", "xml_idx", "Match_Confidence"]
        ]
    )


def count_words(text_series: pd.Series) -> int:
    return sum(len(str(text).split()) for text in text_series)


def calculate_class_metrics(
    extracted_df: pd.DataFrame,
    labelled_pdf_df: pd.DataFrame,
    classified_pdf_df: pd.DataFrame,
    class_num: int,
) -> Dict[str, Any]:
    """
    Calculate metrics for a specific text class.

    Args:
        extracted_df: DataFrame containing extracted labels
        labelled_pdf_df: DataFrame containing PDF labels
        classified_pdf_df: DataFrame containing classified PDF labels
        labelled_pdf_df: DataFrame containing PDF labels
        class_num: Text class number to analyze

    Returns:
        Dictionary containing metrics for the class
    """
    class_name = TextClass(class_num).name

    # Extract records for this class
    extracted_class_df = extracted_df[extracted_df["SourceClass"] == class_num]
    pdf_class_df = labelled_pdf_df[labelled_pdf_df["LabelledClass"] == class_num]
    classified_pdf_df = classified_pdf_df[
        classified_pdf_df["PredictedClass"] == class_num
    ]

    # Calculate metrics
    extracted_records = len(extracted_class_df)
    pdf_records = len(pdf_class_df)
    classified_pdf_records = len(classified_pdf_df)

    # Calculate ratios and averages
    record_ratio = (
        round(pdf_records / extracted_records, 2) if extracted_records > 0 else "nan"
    )
    classified_pdf_ratio = (
        round(classified_pdf_records / extracted_records, 2)
        if extracted_records > 0
        else "nan"
    )
    (
        round(pdf_class_df["Match_Confidence"].mean(), 2)
        if len(pdf_class_df) > 0
        else "nan"
    )

    if class_num == TextClass.PARAGRAPH:
        cont_class = TextClass.PARAGRAPH_CONT
    elif class_num == TextClass.HEADING:
        cont_class = TextClass.HEADING_CONT
    elif class_num == TextClass.BULLET_LIST:
        cont_class = TextClass.BULLET_LIST_CONT
    elif class_num == TextClass.ENUM_LIST:
        cont_class = TextClass.ENUM_LIST_CONT
    else:
        cont_class = None

    if cont_class is not None:
        cont_class_df = labelled_pdf_df[labelled_pdf_df["LabelledClass"] == cont_class]
        num_source_words = count_words(extracted_class_df["text"])
        num_pdf_words = count_words(pdf_class_df["text"]) + count_words(
            cont_class_df["text"]
        )
        word_ratio = (
            round(num_pdf_words / num_source_words, 2)
            if num_source_words > 0
            else "nan"
        )
    else:
        num_source_words = count_words(extracted_class_df["text"])
        num_pdf_words = count_words(pdf_class_df["text"])
        word_ratio = (
            round(num_pdf_words / num_source_words, 2)
            if num_source_words > 0
            else "nan"
        )

    return {
        "SourceClass": class_name,
        "extracted_records": extracted_records,
        "pdf_records": pdf_records,
        "classified_pdf_records": classified_pdf_records,
        # 'avg_confidence': avg_confidence,
        "source_words": num_source_words,
        "pdf_words": num_pdf_words,
        "word_ratio": word_ratio,
        "record_ratio": record_ratio,
        "classified_pdf_ratio": classified_pdf_ratio,
    }


def calculate_all_metrics_dataframe(
    extracted_df: pd.DataFrame,
    labelled_pdf_df: pd.DataFrame,
    classified_pdf_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate metrics for all text classes and return as a pandas DataFrame.

    Args:
        extracted_df: DataFrame containing extracted labels
        labelled_pdf_df: DataFrame containing PDF labels
        classified_pdf_df: DataFrame containing classified PDF labels

    Returns:
        DataFrame containing metrics for all classes
    """
    classes = [
        TextClass.HEADING,
        TextClass.TOC,
        TextClass.PARAGRAPH,
        TextClass.HEADING_CONT,
        TextClass.BULLET_LIST,
        TextClass.ENUM_LIST,
        TextClass.PARAGRAPH_CONT,
        TextClass.BULLET_LIST_CONT,
        TextClass.ENUM_LIST_CONT,
        TextClass.TABLE,
        TextClass.FOOTER,
        TextClass.HEADER,
        TextClass.FOOT_NOTE,
    ]

    metrics_list = []
    for class_num in classes:
        metrics = calculate_class_metrics(
            extracted_df, labelled_pdf_df, classified_pdf_df, class_num
        )
        metrics_list.append(metrics)

    metric_df = pd.DataFrame(metrics_list)

    return metric_df


def print_header() -> None:
    """Print the table header with column names."""
    header = (
        f"{'SourceClass': <{COLUMN_WIDTH}} |  "
        f"{'# XML Records': <{COLUMN_WIDTH}} |  "
        # f"{'# XML Words': <{COLUMN_WIDTH}} |  "
        f"{'# PDF Records': <{COLUMN_WIDTH}} |  "
        f"{'# Classified PDF Records': <{COLUMN_WIDTH}} |  "
        # f"{'# PDF Words': <{COLUMN_WIDTH}} |  "
        f"{'# PDF / XML Records': <{COLUMN_WIDTH}} |  "
        # f"{'Average Match Confidence': <{COLUMN_WIDTH}} |  "
        f"{'# Classified PDF / PDF Records': <{COLUMN_WIDTH}}"
    )
    print(header)
    print("-" * len(header))


def print_row(metrics: Dict[str, Any]) -> None:
    """
    Print a formatted row of metrics.

    Args:
        metrics: Dictionary containing class metrics
    """
    print(
        f"{metrics['SourceClass']: <{COLUMN_WIDTH}} |  "
        f"{metrics['extracted_records']: <{COLUMN_WIDTH}} |  "
        # f"{metrics['extracted_words']: <{COLUMN_WIDTH}} |  "
        f"{metrics['pdf_records']: <{COLUMN_WIDTH}} |  "
        f"{metrics['classified_pdf_records']: <{COLUMN_WIDTH}} |  "
        # f"{metrics['pdf_words']: <{COLUMN_WIDTH}} |  "
        f"{metrics['record_ratio']: <{COLUMN_WIDTH}} |  "
        # f"{metrics['avg_confidence']: <{COLUMN_WIDTH}} |  "
        f"{metrics['classified_pdf_ratio']: <{COLUMN_WIDTH}} |  "
    )


def print_totals(
    extracted_df: pd.DataFrame,
    labelled_pdf_df: pd.DataFrame,
    classified_pdf_df: pd.DataFrame,
) -> None:
    """
    Print the totals row at the bottom of the table.

    Args:
        extracted_df: DataFrame containing extracted labels
        labelled_pdf_df: DataFrame containing PDF labels
    """
    total_extracted_records = len(extracted_df)
    # total_extracted_words = count_words(extracted_df['text'])
    total_pdf_records = len(labelled_pdf_df)
    total_classified_pdf_records = len(classified_pdf_df)
    # total_pdf_words = count_words(labelled_pdf_df['text'])
    total_ratio = (
        round(total_pdf_records / total_extracted_records, 2)
        if total_extracted_records > 0
        else "nan"
    )
    total_classified_pdf_ratio = (
        round(total_classified_pdf_records / total_pdf_records, 2)
        if total_pdf_records > 0
        else "nan"
    )

    print("-" * (COLUMN_WIDTH * 7 + 14))  # Account for separators
    print(
        f"{'Total': <{COLUMN_WIDTH}} |  "
        f"{total_extracted_records: <{COLUMN_WIDTH}} |  "
        # f"{total_extracted_words: <{COLUMN_WIDTH}} |  "
        f"{total_pdf_records: <{COLUMN_WIDTH}} |  "
        f"{total_classified_pdf_records: <{COLUMN_WIDTH}} |  "
        # f"{total_pdf_words: <{COLUMN_WIDTH}} |  "
        f"{total_ratio: <{COLUMN_WIDTH}} |  "
        f"{total_classified_pdf_ratio: <{COLUMN_WIDTH}}"
    )


def get_unused_xml(
    extracted_df: pd.DataFrame, labelled_pdf_df: pd.DataFrame
) -> pd.DataFrame:
    counter = 0
    for idx in range(len(extracted_df)):
        xml_row = extracted_df.iloc[idx]
        xml_idx = xml_row["index"]
        group = labelled_pdf_df[labelled_pdf_df["xml_idx"] == xml_idx]
        class_value = xml_row["SourceClass"]
        if class_value != TextClass.PARAGRAPH:
            continue
        xml_line = xml_row["text"]
        if len(group) == 0:
            print(f"No match found for {xml_idx} {xml_line}")
            continue
        xml_line = clean_text(xml_line)
        matched_texts = []
        pdf_idxs = []
        for i, row in group.iterrows():
            match_text = row["source_matched_text"]
            matched_texts.append(match_text)
            page_number = row["PageNumber"]
            pdf_idxs.append(row["pdf_idx"])

        class_value = group.iloc[0]["ClassLabel"]
        matched_str = " ".join(matched_texts)
        if len(matched_str) < len(xml_line):
            if class_value != TextClass.PARAGRAPH:
                continue
            print(
                f"-------------------------------- {page_number=} --------------------------------"
            )
            print(
                f"{xml_idx=} {len(matched_str)=} {pdf_idxs=} Class: {group.iloc[0]['SourceClass']}"
            )
            print(f"{xml_line   =}")
            print()
            print(f"{matched_str=}")
            print()
            counter += 1
            if counter > 15:
                break


def print_imperfect_matches(labelled_pdf_df: pd.DataFrame) -> None:
    for idx, row in labelled_pdf_df.iterrows():
        if clean_text(row["text"]) != row["source_matched_text"]:
            print(f"{idx=} {row['text']}")
            print(f"{row['source_matched_text']}")
            print()


def generate_metrics_report(
    extracted_labels_path: Path, labelled_pdf_path: Path, classified_pdf_path: Path
) -> pd.DataFrame:
    """
    Generate and display a comprehensive metrics report comparing extracted and PDF labels.

    This function loads the data, calculates metrics for each text class,
    and displays the results in a formatted table.

    Returns:
        DataFrame containing the metrics for all classes
    """
    # Load data
    extracted_df, labelled_pdf_df, classified_pdf_df = load_dataframes(
        extracted_labels_path, labelled_pdf_path, classified_pdf_path
    )

    # print_imperfect_matches(labelled_pdf_df)

    # get_unused_xml(extracted_df, labelled_pdf_df)
    # Print summary information

    print(f"Dataframe: ".ljust(20), extracted_labels_path.stem)
    print(f"Number of Records: ".ljust(20), len(extracted_df))
    print(f"Number of Pages:".ljust(20), len(extracted_df["PageNumber"].unique()))
    print()

    classes = [
        TextClass.PARAGRAPH,
        TextClass.HEADING,
        TextClass.BULLET_LIST,
        TextClass.ENUM_LIST,
        # TextClass.TOC,
        # TextClass.PARAGRAPH_CONT,
        # TextClass.HEADING_CONT,
        # TextClass.BULLET_LIST_CONT,
        # TextClass.ENUM_LIST_CONT,
        # TextClass.TABLE,
        # TextClass.FOOTER,
        # TextClass.HEADER,
        # TextClass.FOOT_NOTE,
    ]

    df = calculate_all_metrics_dataframe(
        extracted_df, labelled_pdf_df, classified_pdf_df
    )
    print(df)
    return df


def save_metrics_dataframe(df: pd.DataFrame, output_path: Path, filename: str) -> None:
    """
    Save a metrics DataFrame to CSV in the metrics subdirectory.

    Args:
        df: DataFrame to save
        output_path: Base path for output
        filename: Name of the file (without extension)
    """
    metrics_dir = output_path / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    csv_path = metrics_dir / f"{filename}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to: {csv_path}")


def combine_all_metrics_csvs(
    metrics_dir: Path, output_filename: str = "combined_metrics.csv"
) -> None:
    """
    Combine all CSV files in the metrics directory into a single CSV file.

    Args:
        metrics_dir: Directory containing individual metrics CSV files
        output_filename: Name of the combined output file
    """
    if not metrics_dir.exists():
        print(f"Metrics directory does not exist: {metrics_dir}")
        return

    csv_files = list(metrics_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in: {metrics_dir}")
        return

    combined_dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Add a column to identify the source file
        df["source_file"] = csv_file.stem
        combined_dfs.append(df)

    if combined_dfs:
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        combined_path = metrics_dir / output_filename
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined {len(csv_files)} CSV files into: {combined_path}")
        print(f"Total rows in combined file: {len(combined_df)}")


if __name__ == "__main__":
    from ai_doc_parser import CFR_PDF as pdf_path
    from ai_doc_parser import CFR_SOURCE as source_path
    from ai_doc_parser import EASA_PDF, EASA_SOURCE
    from ai_doc_parser import LATEX_DIR as document_dir
    from ai_doc_parser import LATEX_SOURCE as source_path

    # Configuration constants
    pdf_paths = list(document_dir.glob("*.pdf"))

    document_dir = Path(
        r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\validation"
    )
    source_path = document_dir / "inspird.tex"
    pdf_paths = list(document_dir.glob("*.pdf"))
    pdf_paths = [EASA_PDF]
    source_path = EASA_SOURCE

    # source_path = Path("/home/rmenon/source/ai-pdf-parser/data/documents/CFR/CFR-2024-title21-vol8-chapI-subchapH.xml")
    source_path = Path(
        r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\validation\CFR-2025-title4-vol1.xml"
    )
    pdf_paths = [
        Path(
            r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\validation\CFR-2025-title4-vol1.pdf"
        ),
        # Path("/home/rmenon/source/ai-pdf-parser/data/documents/CFR/CFR-2024-title21-vol8-chapI-subchapH.pdf"),
    ]
    # pdf_paths = [pdf_path]

    all_metrics_dfs = {}

    for pdf_path in pdf_paths:
        extracted_labels_path = (
            pdf_path.parent / "labelled_source" / f"{source_path.stem}.csv"
        )
        labelled_pdf_path = pdf_path.parent / "labelled_pdf" / f"{pdf_path.stem}.csv"
        classified_pdf_path = (
            pdf_path.parent / "ai_parsed_pdf_no_heuristics" / f"{pdf_path.stem}.csv"
        )
        # ai_results_no_heuristics_path = pdf_path.parent / "ai_parsed_pdf_no_heuristics" / f"{pdf_path.stem}.csv"
        # prepared_features_path = pdf_path.parent / "prepared_features" / f"{pdf_path.stem}.csv"
        # if not labelled_pdf_path.exists():
        # continue

        metrics_df = generate_metrics_report(
            extracted_labels_path, labelled_pdf_path, classified_pdf_path
        )

        # Save individual metrics DataFrame
        # save_metrics_dataframe(metrics_df, document_dir, pdf_path.stem)

        # Store for later combination
        all_metrics_dfs[pdf_path.stem] = metrics_df

    # sort all_metrics_df by classsified_pdf_ratio column paragraph class, descending to reveal the poorest performing classes
    all_metrics_dfs = sorted(
        all_metrics_dfs.items(),
        key=lambda x: x[1].loc[TextClass.PARAGRAPH, "record_ratio"],
        reverse=True,
    )
    for pdf_path, metrics_df in all_metrics_dfs:
        df = pd.read_csv(document_dir / "labelled_pdf" / f"{pdf_path}.csv")
        num_pages = len(df["PageNumber"].unique())

        print(
            f"{metrics_df.loc[TextClass.PARAGRAPH, 'record_ratio']} {num_pages=} {pdf_path=}"
        )

    combine_all_metrics_csvs(document_dir / "metrics")
