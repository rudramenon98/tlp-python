import numpy as np
import pandas as pd
from ai_doc_parser import DATA_DIR
from ai_doc_parser.text_class import TextClass


def main():

    data_dir = DATA_DIR / "documents"
    class_centroid_path = data_dir / "models" / "class_centroid.csv"

    ###################################
    ai_parsed_path = data_dir / "MDR" / "ai_parsed_pdf" / "CELEX_32017R0745_EN_TXT.csv"
    pdf_idx = 153
    ###################################

    df = pd.read_csv(ai_parsed_path)
    ai_row = df[df["pdf_idx"] == pdf_idx].iloc[0]

    centroid_df = pd.read_csv(class_centroid_path)

    # print table of classes, features, and closest distance to the row
    for i, row in centroid_df.iterrows():
        feature_name = row["Feature Name"]
        ai_row_value = ai_row[feature_name]
        row_values = row.iloc[1:]
        classification = ai_row["PredictedClass"]
        closest_feature_index = np.argmin(np.abs(row_values - ai_row_value))
        closest_feature_name = row.index[closest_feature_index + 1]
        centroid_df.loc[i, "Closest Feature"] = closest_feature_name
        centroid_df.loc[i, "AI Row Value"] = ai_row_value
        centroid_df.loc[i, "Classification"] = TextClass(classification).name
        centroid_df.loc[i, "text"] = ai_row["text"]
        centroid_df.loc[i, "pdf_idx"] = ai_row["pdf_idx"]
    print(ai_parsed_path.exists(), ai_parsed_path.parent)
    centroid_df.to_csv(ai_parsed_path.parent / "centroid_analysis.csv", index=False)


if __name__ == "__main__":
    main()
