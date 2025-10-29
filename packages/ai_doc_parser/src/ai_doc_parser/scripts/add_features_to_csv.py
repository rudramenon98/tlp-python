from ai_doc_parser.inference.feature_computation.feature_computer import all_caps, first_word_compound, is_title_case
from typing import Callable
import pandas as pd
from pathlib import Path


def add_feature_to_df(df: pd.DataFrame, column_name: str, feature_function: Callable) -> pd.DataFrame:
    df[column_name] = df.apply(feature_function, axis=1)
    return df


def main():
    csv_paths = []
    csv_paths += list(
        Path(r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\CFR\labelled_pdf").glob("*.csv")
    )
    csv_paths += list(
        Path(r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\EASA\labelled_pdf").glob("*.csv")
    )
    csv_paths += list(
        Path(r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\Latex\labelled_pdf").glob("*.csv")
    )
    csv_paths += list(
        Path(r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\Latex_Bullets\labelled_pdf").glob(
            "*.csv"
        )
    )
    csv_paths += list(
        Path(r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\AC\labelled_pdf").glob("*.csv")
    )
    for i, csv_path in enumerate(csv_paths):
        print(f"Adding feature to {csv_path} {i+1}/{len(csv_paths)}")
        df = pd.read_csv(csv_path)
        df = add_feature_to_df(df, 'is_title_case', is_title_case)
        df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
