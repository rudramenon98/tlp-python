from pathlib import Path
from typing import Any, Dict, List

# Docling imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


def extract_ocr(pdf_path: Path, page_num: int) -> List[Dict[str, Any]]:
    """
    Extract text *blocks* from a single page of a PDF using Docling's OCR.

    This function forces full-page OCR (via the EasyOCR backend) so that
    layout elements are reconstructed even for scanned PDFs.

    Parameters
    ----------
    pdf_path : Path
        Path to a local PDF file.
    page_num : int
        1-based page number to extract from. (Page 1 is the first page.)

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, each representing a Docling text item on the
        requested page. Each dict contains:

        - "text": str — the textual content of the block
        - "label": str — Docling item label (e.g., PARAGRAPH, TITLE, PAGE_HEADER)
        - "bbox": tuple[float, float, float, float] — (x0, y0, x1, y1) in PDF points
        - "page": int — 1-based page number
        - "source": str — short description of how the block was produced

    Notes
    -----
    - Bounding boxes are derived from the Docling provenance for that item and
      unioned across all fragments belonging to the requested page.
    - This function only returns items whose provenance includes the requested page.
    - Requires the Docling package and the chosen OCR backend (EasyOCR here)
      to be installed and available.
    """
    if not isinstance(pdf_path, Path):
        pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if page_num < 1:
        raise ValueError("page_num must be 1-based (>= 1)")

    # --- Configure Docling to run full-page OCR via EasyOCR ---
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    # Force OCR for every page (useful for scanned or tricky PDFs)
    pipeline_options.ocr_options = EasyOcrOptions()

    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

    # Convert the document
    result = converter.convert(str(pdf_path))
    ddoc = result.document

    # Collect text items present on the given page
    blocks: List[Dict[str, Any]] = []

    # Docling exposes text-like items via ddoc.texts (TextItem / Title / Paragraph etc.)
    # Each item has a `.label`, `.text`, and `.prov` list with `page_no` and `bbox`.
    for item in ddoc.texts:
        # Examine provenance fragments that land on the target page
        page_frags = [p for p in (item.prov or []) if getattr(p, "page_no", None) == page_num]
        if not page_frags:
            continue

        # Union all bboxes on this page
        x0 = min(p.bbox.x0 for p in page_frags)
        y0 = min(p.bbox.y0 for p in page_frags)
        x1 = max(p.bbox.x1 for p in page_frags)
        y1 = max(p.bbox.y1 for p in page_frags)

        blocks.append(
            {
                "text": getattr(item, "text", ""),
                "label": getattr(item.label, "name", str(item.label)),
                "bbox": (float(x0), float(y0), float(x1), float(y1)),
                "page": page_num,
                "source": "docling-ocr(easyocr, full-page)",
            }
        )

    return blocks


if __name__ == "__main__":
    pdf_path = Path(
        r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\CFR\CFR-2024-title21-vol8-chapI-subchapH.pdf"
    )
    page_num = 11
    blocks = extract_ocr(pdf_path, page_num)
    print(blocks)
