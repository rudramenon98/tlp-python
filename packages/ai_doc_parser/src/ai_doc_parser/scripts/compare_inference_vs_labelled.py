from pathlib import Path

import pandas as pd

from ai_doc_parser.text_class import TextClass


def main():
    inference_path = Path(
        r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\validation\ai_parsed_pdf_not_combined\CFR-2023-title14-vol5.csv"
    )
    labelled_path = Path(
        r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\validation\labelled_pdf\CFR-2023-title14-vol5.csv"
    )

    output_path = labelled_path.parent.parent / "comparison" / f"{labelled_path.stem}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Create comparison table
    comparison_data = []
    inference_df = pd.read_csv(inference_path)
    labelled_df = pd.read_csv(labelled_path)

    for i, row in labelled_df.iterrows():
        if i % 100 == 0:
            print(f"Processing row {i} of {len(labelled_df)}")
        pdf_idx = row['pdf_idx']

        inference_row = inference_df[inference_df['pdf_idx'] == pdf_idx].iloc[0]

        if inference_row['FinalClass'] != row['LabelledClass']:
            comparison_row = row.to_dict()
            comparison_row.update(inference_row.to_dict())
            comparison_row['LabelledClass'] = TextClass(row['LabelledClass'])
            comparison_row['PredictedClass'] = TextClass(inference_row['PredictedClass'])
            comparison_row['LabelledClassName'] = TextClass(row['LabelledClass']).name
            comparison_row['PredictedClassName'] = TextClass(inference_row['PredictedClass']).name
            comparison_row['FinalClass'] = TextClass(inference_row['FinalClass'])
            comparison_row['FinalClassName'] = TextClass(inference_row['FinalClass']).name
            comparison_data.append(comparison_row)
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_path, index=False)
    print(f"Saved output to {output_path}")
    return comparison_df


if __name__ == "__main__":
    main()
