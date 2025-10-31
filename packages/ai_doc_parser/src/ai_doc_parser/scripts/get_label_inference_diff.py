from pathlib import Path

import pandas as pd
from ai_doc_parser.text_class import TextClass


def main():
    inference_path = Path(
        r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\validation\ai_parsed_pdf_no_heuristics\CFR-2025-title4-vol1.csv"
    )
    labelled_path = Path(
        r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\validation\labelled_pdf\CFR-2025-title4-vol1.csv"
    )
    inference_diff_path = labelled_path.parent.parent / "label_inference_diff" / f"{labelled_path.stem}.csv"
    inference_diff_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving output to {inference_diff_path}")
    inference_df = pd.read_csv(inference_path)
    labelled_df = pd.read_csv(labelled_path)

    output_df = []
    for _, row in inference_df.iterrows():
        pdf_idx = int(row['pdf_idx'])
        labelled_row = labelled_df[labelled_df['pdf_idx'] == pdf_idx]
        if labelled_row.empty:
            continue
        labelled_class = labelled_row['LabelledClass'].values[0]
        inference_class = row['PredictedClass']
        if labelled_class != inference_class:
            row['LabelledClass'] = labelled_class
            row['LabelledClassName'] = TextClass(labelled_class).name
            row.update(labelled_row.iloc[0].to_dict())
            output_df.append(row.to_dict())

    output_df = pd.DataFrame(output_df)
    output_df.to_csv(inference_diff_path, index=False)
    print(f"Saved output to {inference_diff_path}")


if __name__ == "__main__":
    main()
