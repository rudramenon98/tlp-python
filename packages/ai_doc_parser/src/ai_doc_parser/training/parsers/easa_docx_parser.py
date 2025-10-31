import logging
import time
import traceback
from pathlib import Path

import pandas as pd
from ai_doc_parser.training.parsers.base_parser import TextClass
from lxml import etree as ET

log = logging.getLogger(__name__)


def detect(text):
    """Simple text validation function to check if text is valid."""
    if not text:
        return False
    text = str(text).strip()
    return len(text) > 0 and not text.isspace()


def easa_doc_parsing(path: Path | str) -> pd.DataFrame:
    start = time.time()
    tree = ET.parse(path)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    # Initialize dictionary to store all data
    data_dict = {
        "LineNumbers": [],
        "text": [],
        "PageNumber": [],
        "Class": [],
        "ClassName": [],
    }

    def append_to_dict(
        elem,
        text,
        class_val: TextClass,
    ):
        """Helper function to append data to the dictionary"""
        data_dict["LineNumbers"].append(elem.sourceline)
        data_dict["text"].append(text)
        data_dict["PageNumber"].append(1)  # Default page number for EASA docs
        data_dict["Class"].append(class_val.value)
        data_dict["ClassName"].append(class_val.name.replace("_", " ").title())

    for i in tree.iter():
        try:
            style = i.find(".//w:pStyle", ns)

            if style is not None:
                # Check if i has a tag attribute before accessing it
                if not hasattr(i, "tag"):
                    continue
                # Check if tag is a string before calling replace
                if not isinstance(i.tag, str):
                    continue
                tag = i.tag.replace(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}", ""
                )
                check_tag = style.attrib[
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
                ]

                if (
                    str(check_tag).lower().startswith("heading")
                    or "heading" in str(check_tag).lower()
                ):
                    if tag == "p":
                        headingext = "".join(i.itertext())
                        if headingext != "" and detect(headingext):
                            append_to_dict(i, headingext, TextClass.HEADER)

                elif str(check_tag).lower().startswith("table"):
                    if tag == "p":
                        table_text = "".join(i.itertext())
                        if table_text != "" and detect(table_text):
                            append_to_dict(i, table_text, TextClass.TABLE)

                elif str(check_tag).lower().startswith("toc"):
                    if tag == "p":
                        for j in i.iter():
                            # Check if j has a tag attribute before accessing it
                            if not hasattr(j, "tag"):
                                continue
                            # Check if tag is a string before calling replace
                            if not isinstance(j.tag, str):
                                continue
                            toctag = j.tag.replace(
                                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}",
                                "",
                            )

                            if toctag == "p":
                                toctext = "".join(j.itertext()).split()
                                toctext = "  ".join(
                                    [
                                        i
                                        for i in toctext
                                        if not i.startswith("\\")
                                        and "PAGEREF" != i
                                        and not i.startswith("_Toc")
                                    ]
                                )
                                if toctext != "" and detect(toctext):
                                    append_to_dict(j, toctext, TextClass.TOC)
                else:
                    if tag == "p":
                        paragraphtext = "".join(i.itertext())
                        if str(paragraphtext) != "":
                            log.debug("Text", paragraphtext)
                            if detect(paragraphtext):
                                append_to_dict(i, paragraphtext, TextClass.NORMAL_TEXT)

            else:
                # Check if i has a tag attribute before accessing it
                if not hasattr(i, "tag"):
                    continue
                # Check if tag is a string before calling replace
                if not isinstance(i.tag, str):
                    continue
                tag = i.tag.replace(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}", ""
                )
                if tag == "p":
                    paragraphtext = "".join(i.itertext())
                    if paragraphtext != "":
                        if detect(paragraphtext):
                            log.debug("paraptext", paragraphtext)
                            append_to_dict(i, paragraphtext, TextClass.NORMAL_TEXT)

        except Exception as e:
            log.error(f"Error parsing EASA document: {e}")
            traceback.print_exc()
            continue

    # Create DataFrame from collected data
    raw_form = pd.DataFrame(data_dict)
    raw_form = raw_form.dropna()

    end = time.time()
    log.debug("Time for EASA XML parsing = " + str(end - start))
    log.debug("SUCCESS")

    # use only the requested columns
    raw_form = raw_form[["LineNumbers", "text", "PageNumber", "Class", "ClassName"]]
    return raw_form


def split_paragraphs_into_chunks(text, n_words):

    # split the text
    pieces = text.split()

    # return the chunks
    return list(
        " ".join(pieces[i : i + n_words]) for i in range(0, len(pieces), n_words)
    )


def main():
    data_dir = Path(__file__).parents[4] / "data" / "documents" / "EASA"
    xml_path = data_dir / "Easy Access Rules for Aerodromes _PDF_.xml"
    output_dir = data_dir / "parsed_xml"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{xml_path.stem}.csv"

    print(output_file, output_file.exists())
    df = easa_doc_parsing(xml_path)
    df.to_csv(output_file, index=False)

    print(df.head())


if __name__ == "__main__":
    main()
