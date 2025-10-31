import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from ai_doc_parser.text_class import TextClass
from docx import Document
from docx.table import Table

# ----------------------------
# Helpers
# ----------------------------


def _iter_block_items(parent):
    """
    Yield paragraphs and tables in document order for a given parent (document, header, footer, cell).
    Based on the underlying XML order (w:p or w:tbl).
    """
    parent_elm = parent._element
    for child in (
        parent_elm.body.iterchildren()
        if hasattr(parent_elm, "body")
        else parent_elm.iterchildren()
    ):
        if child.tag.endswith("}p"):  # paragraph
            # Count paragraphs before this one to get the correct index
            para_count = 0
            for sibling in child.itersiblings(preceding=True):
                if sibling.tag.endswith("}p"):
                    para_count += 1
            yield parent.paragraphs[para_count]
        elif child.tag.endswith("}tbl"):
            # find the matching Table object by index
            # python-docx doesn't map elements directly, so we reconstruct via Table wrappers
            # Simpler approach: create a Table wrapper and iterate its cells' paragraphs
            tbl = Table(child, parent)
            yield tbl


def _iter_paragraphs_from_block(block):
    if isinstance(block, Table):
        for row in block.rows:
            for cell in row.cells:
                # For each cell, yield paragraphs and tables (nested tables rare but possible)
                for inner in _iter_block_items(cell):
                    if isinstance(inner, Table):
                        yield from _iter_paragraphs_from_block(inner)
                    else:
                        yield inner
    else:
        yield block  # it is a paragraph


def _is_heading_paragraph(paragraph):
    """
    Heuristic: treat built-in Heading styles (and custom styles starting with 'Heading') as headings.
    """
    style = paragraph.style
    if style is None:
        return False
    name = (style.name or "").lower()
    return name.startswith("heading")


def _paragraph_has_toc_field(paragraph):
    """
    Detects if a paragraph contains a Word TOC field.
    In DOCX, TOC appears as a field with an instruction text run containing 'TOC'.
    """
    # Access the underlying XML runs to look for w:instrText containing 'TOC'
    # Use iterdescendants to find all descendant elements with 'instrText' tag
    for elem in paragraph._p.iterdescendants():
        if elem.tag.endswith("}instrText") and "TOC" in (elem.text or ""):
            return True
    return False


def _classify_paragraph(
    paragraph, scope
) -> TextClass:  # pyright: ignore[reportUndefinedVariable]
    """
    scope: 'body' | 'header' | 'footer'
    Returns one of: TextClass.HEADING, TextClass.HEADER, TextClass.FOOTER, TextClass.TOC, TextClass.PARAGRAPH
    """
    text = (paragraph.text or "").strip()
    if not text:
        return None  # skip empty

    # Header/Footer take precedence if we're in those scopes
    if scope == "header":
        return TextClass.HEADER
    if scope == "footer":
        return TextClass.FOOTER

    # Body scope: check TOC first, then Heading
    if _paragraph_has_toc_field(paragraph):
        return TextClass.TOC
    if _is_heading_paragraph(paragraph):
        return TextClass.HEADING
    return TextClass.PARAGRAPH


# ----------------------------
# Main function
# ----------------------------


def extract_docx(docx_path):
    """
    Build a pandas DataFrame with the SAME schema as produced by `pymu_extract_pdf_text`.

    Notes:
    - DOCX files don't expose page geometry; PDF-specific fields are filled with defaults.
    - Coordinates and sizes are set to NaN; booleans and lists use safe defaults.
    """
    doc = Document(docx_path)
    if not docx_path.exists():
        raise FileNotFoundError(f"File {docx_path} does not exist")
    if not docx_path.suffix == ".docx":
        raise ValueError(f"File {docx_path} is not a DOCX file")
    data_rows: List[Dict] = []

    # Helper to append a row in the target schema
    def append_row(text_value: str, page_number: int, class_value: TextClass):
        text_value = (text_value or "").strip()
        if not text_value:
            return

        row_data = {
            "origin_text": text_value,
            "text": text_value,
            "SourceClass": class_value.value,
            "SourceClassName": class_value.name,
            "x0": float("nan"),
            "y0": float("nan"),
            "x1": float("nan"),
            "y1": float("nan"),
            "block_x0": float("nan"),
            "block_y0": float("nan"),
            "block_x1": float("nan"),
            "block_y1": float("nan"),
            "line_x0": float("nan"),
            "line_y0": float("nan"),
            "line_x1": float("nan"),
            "line_y1": float("nan"),
            "PageNumber": page_number,
            "page_height": float("nan"),
            "page_width": float("nan"),
            "major_font_family": None,
            "major_font_size": None,
            "major_color": "#000000",
            "font_family_info": {},
            "font_size_info": {},
            "color_info": {},
            "multiple_font_family": 0,
            "multiple_font_size": 0,
            "table_coordinates": [],
            "horizontal_lines": [],
            "vertical_lines": [],
            "table_continued_to_next_page": False,
            "is_table": False,
            "crop_x0": 0.0,
            "crop_x1": 0.0,
            "crop_y0": 0.0,
            "crop_y1": 0.0,
            "version": Path(docx_path).stem,
            "ncols": 1,
            "column_borders": [],
            "left_column_blocks": 0,
            "right_column_blocks": 0,
            "left_column_text": 0,
            "right_column_text": 0,
        }

        data_rows.append(row_data)

    # 1) Headers/Footers
    for sec in doc.sections:
        hdr = sec.header
        for block in _iter_block_items(hdr):
            for para in _iter_paragraphs_from_block(block):
                klass = _classify_paragraph(para, scope="header")
                if klass is not None:
                    append_row(para.text, 0, klass)

        ftr = sec.footer
        for block in _iter_block_items(ftr):
            for para in _iter_paragraphs_from_block(block):
                klass = _classify_paragraph(para, scope="footer")
                if klass is not None:
                    append_row(para.text, 0, klass)

    # 2) Body: paragraphs and tables in document order
    for block in _iter_block_items(doc):
        for para in _iter_paragraphs_from_block(block):
            klass = _classify_paragraph(para, scope="body")
            if klass is not None:
                append_row(para.text, 0, klass)

    df = pd.DataFrame(data_rows)
    if not df.empty:
        df = df.reset_index(drop=True)
        df["xml_idx"] = df.index
    return df


# ----------------------------
# Example
# ----------------------------
# results = classify_docx_lines("input.docx")
# for item in results:
#     print(item['class'], "=>", item['text'])

# Example usage
if __name__ == "__main__":
    from ai_doc_parser import EASA_DIR

    logging.basicConfig(level=logging.DEBUG)

    input_path = (
        EASA_DIR.parent / "AUSMDR" / "clinical-evidence-guidelines-medical-devices.docx"
    )
    output_dir = input_path.parent / "docx_extracted"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.csv"
    df = extract_docx(input_path)
    print(df.head())
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        logging.warning("Could not write CSV: %s", e)
