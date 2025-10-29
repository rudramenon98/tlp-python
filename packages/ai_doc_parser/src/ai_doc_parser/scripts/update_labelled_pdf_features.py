from pathlib import Path
import pandas as pd


def update_labelled_pdf_features(feature_df_path: Path, labelled_pdf_path: Path):
    """Get all the values in the feature df and update the labelled pdf df row with the same pdf_idx.
    Do not update any other columns in the labelled pdf df."""

    feature_df = pd.read_csv(feature_df_path)
    labelled_pdf_df = pd.read_csv(labelled_pdf_path)

    # Fast, vectorized index-aligned update on pdf_idx
    if "pdf_idx" not in feature_df.columns or "pdf_idx" not in labelled_pdf_df.columns:
        raise KeyError("Both dataframes must contain a 'pdf_idx' column")

    feature_df = feature_df.set_index("pdf_idx")
    labelled_pdf_df = labelled_pdf_df.set_index("pdf_idx")

    # Ensure all feature columns exist on labelled df (to match original behavior which created new columns)
    missing_cols = [c for c in feature_df.columns if c not in labelled_pdf_df.columns]
    for col in missing_cols:
        labelled_pdf_df[col] = pd.NA

    # Only update rows that exist in both dataframes (do not add new rows)
    common_idx = feature_df.index.intersection(labelled_pdf_df.index)
    if len(common_idx) == 0:
        labelled_pdf_df.reset_index().to_csv(labelled_pdf_path, index=False)
        return

    labelled_pdf_df.loc[common_idx, feature_df.columns] = feature_df.loc[common_idx, feature_df.columns]

    labelled_pdf_df.reset_index().to_csv(labelled_pdf_path, index=False)


def main():
    from ai_doc_parser import EASA_PDF

    document_dir = Path(r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents")
    csv_paths = []
    csv_paths += list((document_dir / "Latex" / "labelled_pdf").glob("*.csv"))
    csv_paths += list((document_dir / "Bullets" / "labelled_pdf").glob("*.csv"))
    # csv_paths += list((document_dir / "CFR" / "labelled_pdf").glob("*.csv"))
    # csv_paths += list((document_dir / "EASA" / "labelled_pdf").glob("*.csv"))
    # csv_paths += list((document_dir / "AUSMDR" / "labelled_pdf").glob("*.csv"))
    # csv_paths += list((document_dir / "validation" / "labelled_pdf").glob("*.csv"))
    for i, csv_path in enumerate(csv_paths):
        print(f"Updating labelled pdf features for {csv_path} {i+1}/{len(csv_paths)}")
        feature_df_path = csv_path.parent.parent / "computed_features" / f"{csv_path.stem}.csv"
        labelled_pdf_path = csv_path
        update_labelled_pdf_features(feature_df_path, labelled_pdf_path)


if __name__ == "__main__":
    main()
