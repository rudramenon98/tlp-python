import pandas as pd


def get_labelled_classified_diff(
    classified_df: pd.DataFrame, labelled_df: pd.DataFrame
) -> pd.DataFrame:
    # Merge the dataframes on pdf_idx to get matching rows efficiently
    merged_df = classified_df.merge(
        labelled_df, on="pdf_idx", how="left", suffixes=("", "_labelled")
    )

    # drop classes that have an ExtractedClass that is not nan
    merged_df = merged_df[merged_df["ExtractedClass"].isna()]

    # Filter rows where predicted_class != ClassLabel or where no match was found
    mask = merged_df["FinalClass"] != merged_df["LabelledClass"]

    # Get the filtered results
    diff_df = merged_df[mask].copy()

    # Clean up duplicate columns (remove _labelled suffix columns)
    labelled_cols = [col for col in diff_df.columns if col.endswith("_labelled")]
    for col in labelled_cols:
        base_col = col.replace("_labelled", "")
        if base_col not in classified_df.columns:
            # Use the labelled value where the base column is missing
            diff_df[base_col] = diff_df[col]
        diff_df = diff_df.drop(columns=[col])

    return diff_df


def main() -> None:
    from ai_doc_parser import EASA_DIR, EASA_PDF

    EASA_PDF = EASA_PDF.parent / (
        "Easy Access Rules for Acceptable Means of Compliance for Airworthiness of Products_ Parts and Appliances _AMC_20_ Amdt 23 _PDF_.pdf"
    )

    pdf_paths = list(EASA_DIR.glob("*.pdf"))[:1]
    pdf_paths = [EASA_PDF]

    for pdf_path in pdf_paths:
        classified_path = pdf_path.parent / "ai_parsed_pdf" / f"{pdf_path.stem}.csv"
        labelled_path = pdf_path.parent / "labelled_pdf" / f"{pdf_path.stem}.csv"

        classified_df = pd.read_csv(classified_path)
        labelled_df = pd.read_csv(labelled_path)
        diff_df = get_labelled_classified_diff(classified_df, labelled_df)

        output_path = (
            pdf_path.parent / "classified_labelled_diff" / f"{pdf_path.stem}.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        diff_df.to_csv(output_path, index=False)

        print(diff_df)


if __name__ == "__main__":
    main()
