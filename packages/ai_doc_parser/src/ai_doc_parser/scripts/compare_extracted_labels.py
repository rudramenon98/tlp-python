from pathlib import Path
import pandas as pd


def main():
    data_dir = Path(__file__).parents[3] / "data" / "documents" / "Latex_Bullets"
    extracted_labels_path = data_dir / "labelled_source" / "inspird_bullets.csv"

    labelled_pdf_path = data_dir / "labelled_pdf" / "Inspird-bullets-left-single.csv"

    extracted_df = pd.read_csv(extracted_labels_path)
    labelled_df = pd.read_csv(labelled_pdf_path)

    line_numbers = extracted_df["LineNumbers"]
    labelled_line_numbers = labelled_df["XML_line_Number"]

    summary_output = ""
    output = ""
    for i, row in extracted_df.iterrows():
        extracted_line_number = row["LineNumbers"]
        xml_text = row["text"]
        block_df = labelled_df[labelled_df["XML_line_Number"] == extracted_line_number]
        pdf_text_block = ""
        for j, block_row in block_df.iterrows():
            pdf_text_block = (
                pdf_text_block
                + f"{block_row['pdf_idx']:>6}"
                + " | "
                + f"{round(block_row['Match_Confidence'], 2):>6}"
                + " | "
                + block_row["text"]
                + " \n"
                + (" " * len("PDF_text: "))
            )
        pdf_text = " ".join([block_row["text"] for i, block_row in block_df.iterrows()])
        percent_words = len(pdf_text.split()) / len(xml_text.split()) * 100
        summary_output = summary_output + f"{extracted_line_number:>6} | {percent_words:>6.2f}%\n"

        output = output + f"--------------{extracted_line_number}--------------\n"
        output = output + f"XML_text: {xml_text}\n"
        output = output + f"PDF_text: {pdf_text_block}\n"

    output = summary_output + "\n" + output
    with open(data_dir / "labelled_pdf" / "output.txt", "w") as f:
        f.write(output)
    print(f"Saved output to {data_dir / 'labelled_pdf' / 'output.txt'}")


if __name__ == "__main__":
    main()
