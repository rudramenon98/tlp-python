import re
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from ai_doc_parser.pdf_extractor.feature_methods import Symbols

# Check if the line contains the title of the document


def hasTOC(line):
    n = line.find("\\ttableofcontents")
    if n < 0:
        return False
    return True


def gather_header_targetlabels(text):
    tag_name = OrderedDict()
    try:

        hdr = []
        pref_list = [
            "\\subsection",
            "\\subsubsection",
            "\\section",
            "\\title",
            "\\subsection",
            "\\subsubsection",
            "\\section",
            "\\title",
        ]

        flag = True
        start_line = ""
        line_number = 0
        for line in text:
            print(">>>>", line)
            line_number = line_number + 1
            if flag == False:
                if "}" in line:
                    start_line = start_line + line
                    tag_name[line_number] = start_line
                    hdr.append(1)
                    flag = True
                    start_line = ""
                else:
                    start_line = start_line + line

            if flag == True:
                if list(filter(line.startswith, pref_list)) != [] or any(
                    line.strip().startswith(ext) for ext in pref_list
                ):
                    if "}" in line:
                        tag_name[line_number] = line
                        hdr.append(1)
                        flag = True
                        start_line = ""
                    else:
                        start_line = start_line + line
                        flag = False

        df = pd.DataFrame(
            {
                "lineNumber": tag_name.keys(),
                "text": tag_name.values(),
                "header_label": hdr,
            }
        )
        return tag_name
    except:
        df = pd.DataFrame({"text": []})
        return tag_name


def remove_latex_environments_(text_line, especific_list):
    for especific in especific_list:
        text_line = str(text_line).replace(especific, "")
    return text_line


def remove_latex_commands_(text_line, especific_list):
    all_founded = []
    for (
        especific
    ) in (
        especific_list
    ):  # latex_commands+latex_math_commands+latex_symbols+paragraph_break+breaks:
        TEXTO = especific
        try:
            re_finds = re.findall(rf"\{TEXTO}", text_line)
        except:
            re_finds = re.findall(rf"{TEXTO}", text_line)
        for finded in re_finds:
            all_founded.append(finded)

    all_founded = sorted(
        list(set(all_founded)), key=len, reverse=True
    )  # sorted based on length

    not_all_char = []
    atleast_one_char = []
    for allfounded in all_founded:
        if any(char in Symbols().one_letter for char in allfounded):
            atleast_one_char.append(allfounded)
        else:
            not_all_char.append(allfounded)

    for founded in atleast_one_char + not_all_char:
        text_line = str(text_line).replace(founded, "")
    return text_line


def accent_and_badtext_handler(badtext):

    try:
        encoded = str(badtext).encode("cp1252")
        goodtext = encoded.decode("utf-8")
    except:
        accented_string = str(badtext)
        badtext = unidecode.unidecode(accented_string)
        encoded = badtext.encode("cp1252")
        goodtext = encoded.decode("utf-8")

    return goodtext


def gather_paragraph_targetlabels_from_latexfile(
    latex_file_path: Path | str, text: str
) -> tuple[list[int], pd.DataFrame]:
    """
    Parse LaTeX file to extract paragraphs and classify them with target labels.

    Args:
        latex_file_path: Path to the LaTeX file
        text: Content of the LaTeX file as string

    Returns:
        Tuple of (header_labels, DataFrame) with classified paragraphs
    """
    # Validate the latex_file_path
    if isinstance(latex_file_path, str):
        latex_file_path = Path(latex_file_path)
    if not latex_file_path.suffix == ".tex":
        raise ValueError(f"LaTeX file path must end with .tex: {latex_file_path}")
    if not latex_file_path.exists():
        raise FileNotFoundError(f"LaTeX file does not exist: {latex_file_path}")

    # Define LaTeX commands that indicate headers or special content
    pref_list = [
        "\\subsection",
        "\\subsubsection",
        "\\section",
        "\\title",
        "\\subsection",
        "\\subsubsection",
        "\\section",
        "\\title",
        "\\begin{table",
        "\\end{table",
    ]
    line_break_patterns = [
        r"(\n+\s*%?\s*\n+)",
        r"(\\par)",
        r"(\\indent)",
        r"(\\noindent)",
        r"(\noindent)",
        r"(\\\\\\)",
        r"(bullet)",
        r"(\\item)",
    ]
    try:
        # Step 1: Extract header information from the document
        hdrs = gather_header_targetlabels(text.splitlines())

        # Step 2: Split text into paragraphs using regex patterns
        # Split on: multiple newlines, LaTeX paragraph commands, line breaks, bullets, list items
        line_break_pattern = "|".join(line_break_patterns)
        paragraphs = [
            par
            for par in re.split(line_break_pattern, text)
            if par != None and len(par.strip()) > 0
        ]

        # Step 3: Process LaTeX environments (like tables) by splitting around begin/end commands
        hdr_labels = []
        lastHeaderMatch = 0
        new_paragraphs = []
        for paragraph in paragraphs:
            lines = paragraph.split("\n")
            pars = []
            pars_end = []

            # Debug print for table content
            if "begin{tabl" in paragraph:
                print()

            # Handle \begin{environment} commands - split content around them
            if "begin{" in paragraph:
                # Find all lines containing \begin{ commands
                indx = [i for i, line in enumerate(lines) if "begin{" in line]
                # Process from end to beginning to maintain indices
                for ind in indx[::-1]:
                    if ind + 1 >= len(lines):
                        continue
                    if lines[ind + 1].startswith("\\"):
                        continue
                    # Split paragraph around the \begin{ command
                    p1 = "\n".join(lines[:ind])
                    p2 = "\n".join(lines[ind + 1 :])
                    pars = [p1, p2]

            # Handle \end{environment} commands - split content around them
            if "end{" in paragraph:
                if len(pars) > 0:
                    # Process existing splits further
                    for par_ in pars:
                        pars_end = []
                        lines = par_.split("\n")
                        indx = [_ for _, line in enumerate(lines) if "end{" in line]
                        for ind in indx[::-1]:
                            if ind - 1 <= -1:
                                continue
                            # Skip if previous line is also a LaTeX command
                            if lines[ind - 1].startswith("\\"):
                                pars_end.append(par_)
                                continue
                            else:
                                # Split paragraph around the \end{ command
                                p1 = "\n".join(lines[:ind])
                                p2 = "\n".join(lines[ind + 1 :])
                                pars_end.append(p1)
                                pars_end.append(p2)

                if len(pars) == 0:
                    # Handle \end{ commands in original paragraph
                    indx = [_ for _, line in enumerate(lines) if "end{" in line]
                    for ind in indx[::-1]:
                        if ind - 1 <= -1:
                            continue
                        if lines[ind - 1].startswith("\\"):
                            continue
                        p1 = "\n".join(lines[:ind])
                        p2 = "\n".join(lines[ind + 1 :])
                        pars = [p1, p2]

            # Add processed paragraphs to new list
            if len(pars_end) > 0:
                for par_ in pars_end:
                    new_paragraphs.append(par_)
            elif len(pars) > 0:
                for par_ in pars:
                    new_paragraphs.append(par_)
            else:
                new_paragraphs.append(paragraph)
        paragraphs = new_paragraphs

        # Step 4: Remove \section commands from paragraph content
        new_paragraphs = []
        for paragraph in paragraphs:
            if "\\section" in paragraph:
                indx = [_ for _, line in enumerate(lines) if "\\section" in line]
                for ind in indx[::-1]:
                    if ind + 1 < len(lines):
                        if lines[ind + 1].startswith("\\"):
                            continue
                        else:
                            # Remove the \section line from paragraph
                            del lines[ind]
                            paragraph = "\n".join(lines)
                    else:
                        continue
            new_paragraphs.append(paragraph)
        paragraphs = new_paragraphs

        # Step 5: Clean paragraphs by removing LaTeX commands and comments
        new_paragraphs = []
        for paragraph in paragraphs:
            if "table" in paragraph:
                print()

            # Remove \label commands
            if "\\label" in paragraph:
                indx = [
                    _
                    for _, line in enumerate(lines)
                    if line.strip().startswith("\\label")
                ]
                for ind in indx[::-1]:
                    p_list = paragraph.split("\n")
                    del p_list[ind]
                    paragraph = "\n".join(p_list)

            # Remove comment lines (starting with %)
            if "%" in paragraph:
                indx = [
                    _
                    for _, line in enumerate(paragraph.split("\n"))
                    if line.strip().startswith("%")
                ]
                for ind in sorted(indx, reverse=True):
                    p_list = paragraph.split("\n")
                    del p_list[ind]
                    paragraph = "\n".join(p_list)

            # Remove other LaTeX commands (but keep headers and author info)
            if "\\" in paragraph:
                indx = [
                    _
                    for _, line in enumerate(paragraph.split("\n"))
                    if line.strip().startswith("\\")
                    and not line.strip().startswith("\\author")
                    and not any(line.strip().startswith(ext) for ext in pref_list)
                ]
                for ind in sorted(indx, reverse=True):
                    p_list = paragraph.split("\n")
                    del p_list[ind]
                    paragraph = "\n".join(p_list)

            # Only keep non-empty paragraphs
            if len(paragraph.strip()) != 0:
                new_paragraphs.append(paragraph)
                if "\\begin{table" in paragraph:
                    print("table start")
                if "\\end{table" in paragraph:
                    print("table end")

                # Step 6: Classify paragraphs as headers or not
                isHeader = False
                for i, hdr in enumerate(hdrs.values()):
                    if hdr == paragraph and i >= lastHeaderMatch + 1:
                        # Mark as header
                        hdr_labels.append(1)
                        isHeader = True
                        lastHeaderMatch = i
                        break

                if not isHeader:
                    # Mark as regular paragraph
                    hdr_labels.append(0)
        paragraphs = new_paragraphs

        # Step 7: Generate page labels (dummy values for PDF compatibility)
        para_labels = []
        page_labels = []
        for i in hdr_labels:
            # Dummy page number as the PDF matcher needs it
            page_labels.append("1")
            if i == 1:
                para_labels.append(0)
            else:
                para_labels.append(1)

        # Step 8: Generate table labels by tracking table environments
        table_labels = []
        inTable = False
        for paragraph in paragraphs:
            if "\\begin{table" in paragraph:
                inTable = True
                table_labels.append(1)
                continue
            elif "\\end{table" in paragraph:
                inTable = False
                table_labels.append(1)
                continue
            if inTable:
                # Content inside table
                table_labels.append(1)
            else:
                # Content outside table
                table_labels.append(0)

        # Step 9: Clean and normalize paragraph text
        new_paragraphs = []
        for paragraph in paragraphs:
            # Remove LaTeX commands, environments, and symbols, then handle accents
            clean_paragraph = remove_latex_environments_(
                remove_latex_commands_(
                    paragraph.lower(),
                    Symbols().latex_commands
                    + Symbols().latex_math_commands
                    + Symbols().latex_next_line
                    + Symbols().latex_symbols,
                ),
                Symbols().latex_environments,
            )
            new_paragraphs.append(accent_and_badtext_handler(clean_paragraph))
        paragraphs = new_paragraphs

        # Step 10: Create final DataFrame with all labels
        df = pd.DataFrame(
            {
                "correct_raw_text": paragraphs,
                "page": page_labels,
                "header_target": hdr_labels,
                "table_target": table_labels,
                "toc_target": [np.nan] * len(table_labels),
            }
        )
        return hdr_labels, df

    except:
        # Error handling - return empty DataFrame
        traceback.print_exc()
        df = pd.DataFrame({"text": []})
        return hdr_labels, df.reset_index(drop=True)


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame):
    if df1.shape != df2.shape:
        print("DataFrames have different shapes:")
        print(f"df1 shape: {df1.shape}, df2 shape: {df2.shape}")
    diffs = []
    for i in range(len(df1)):
        if df1.iloc[i]["correct_raw_text"] != df2.iloc[i]["correct_raw_text"]:
            print(f"Difference at index {i}:")
            print(f"Old: {df1.iloc[i]['correct_raw_text']}")
            print(f"New: {df2.iloc[i]['correct_raw_text']}")
            print("-" * 50)
            diffs.append(
                (i, df1.iloc[i]["correct_raw_text"], df2.iloc[i]["correct_raw_text"])
            )

    if not diffs:
        print("DataFrames are identical.")
    else:
        print("Differences found:")
        for index, column, val1, val2 in diffs:
            print(f"At index '{index}', column '{column}': df1 = {val1}, df2 = {val2}")


if __name__ == "__main__":
    data_dir = Path(__file__).parents[4] / "data" / "documents"
    latex_file_path = data_dir / "Latex" / "inspird.tex"
    output_dir = latex_file_path.parent / "parsed_latex"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(latex_file_path, "r") as f:
        text = f.read()
    output_file = (output_dir / latex_file_path.name).with_suffix(".csv")

    hdr_labels, df = gather_paragraph_targetlabels_from_latexfile(latex_file_path, text)

    # Save the dataframe
    df.to_csv(output_file, index=False)
    print(f"Saved parsed data to: {output_file}")
