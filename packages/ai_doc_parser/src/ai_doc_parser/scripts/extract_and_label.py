#!/usr/bin/env python3
"""
PDF and XML Processing Pipeline Script

This script provides a complete pipeline for processing PDF and XML files:
1. Extract PDF using pymu_extractor
2. Compute features on the extracted data
3. Parse XML and create labeled data
4. Apply labeling algorithm to match PDF and XML data
5. Save results in organized output directories

Usage:
    python extract_and_label.py
"""

import logging
import sys
import traceback
from pathlib import Path
from typing import Callable, Tuple

import pandas as pd
from ai_doc_parser import DATA_DIR, EASA_DIR
from ai_doc_parser.inference.ai_pdf_parser import parse_pdf_ai
from ai_doc_parser.inference.feature_computation.feature_computer import (
    compute_features,
)
from ai_doc_parser.inference.pdf_extraction.docx_extactor import extract_docx
from ai_doc_parser.inference.pdf_extraction.pymu_extractor import extract_pdf_text
from ai_doc_parser.scripts.update_labelled_pdf_features import (
    update_labelled_pdf_features,
)
from ai_doc_parser.training.classifier_trainer import load_model
from ai_doc_parser.training.label_extractors.easa_docx_extractor import extract_easa_xml
from ai_doc_parser.training.label_extractors.latex_arvix_extractor import LatexParser
from ai_doc_parser.training.label_extractors.xml_cfr_extractor import cfr_extracting
from ai_doc_parser.training.labeller import sliding_window_labeling
from ai_doc_parser.training.post_labelling_heuristics import (
    apply_heuristic_to_dir,
    latex_heuristics,
)

log = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('extract_and_label.log')],
    )


def create_output_directories(output_dir: Path) -> Tuple[Path, Path, Path, Path]:
    """
    Create output directories for different stages of processing.

    Args:
        output_dir: Base output directory

    Returns:
        Tuple of paths for pdf_extracted, computed_features, labelled_source, labelled_pdf
    """
    pdf_extracted_dir = output_dir / "pdf_extracted"
    computed_features_dir = output_dir / "computed_features"
    labelled_source_dir = output_dir / "labelled_source"
    labelled_pdf_dir = output_dir / "labelled_pdf"
    ai_parsed_dir = output_dir / "ai_parsed_pdf"

    # Create directories if they don't exist
    for dir_path in [pdf_extracted_dir, computed_features_dir, labelled_source_dir, labelled_pdf_dir, ai_parsed_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created/verified directory: {dir_path}")

    return pdf_extracted_dir, computed_features_dir, labelled_source_dir, labelled_pdf_dir, ai_parsed_dir


def extract_and_label(
    pdf_path: Path,
    xml_path: Path,
    pdf_type: str,
    document_dir: Path,
    parsing_method: Callable[[Path], pd.DataFrame],
    overwrite: bool = False,
) -> None:
    """Main function to run the complete pipeline."""
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    # Define paths here - modify these as needed
    # document_dir = Path(__file__).parents[3] / "data" / "documents" / "EASA"
    # pdf_path = document_dir / "Easy Access Rules for Aerodromes _PDF_.pdf"
    # xml_path = document_dir / "Easy Access Rules for Aerodromes _PDF_.xml"
    # pdf_type = "EASA"  # Options: "CFR", "FAA_Advisory_Circulars_Data", "Arxiv"

    # Validate input files
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)

    if not xml_path.exists():
        logger.error(f"XML file not found: {xml_path}")
        sys.exit(1)

    try:
        # Create output directories
        (pdf_extracted_dir, computed_features_dir, labelled_source_dir, labelled_pdf_dir, ai_parsed_dir) = (
            create_output_directories(document_dir)
        )
        pdf_name = pdf_path.stem
        extracted_pdf_path = pdf_extracted_dir / f"{pdf_name}.csv"
        xml_labelled_df_path = labelled_source_dir / f"{xml_path.stem}.csv"
        feature_df_path = computed_features_dir / f"{pdf_name}.csv"
        final_labeled_df_path = labelled_pdf_dir / f"{pdf_name}.csv"
        ai_parsed_df_path = ai_parsed_dir / f"{pdf_name}.csv"
        ai_results_no_heuristics_df_path = ai_parsed_dir.parent / "ai_parsed_pdf_no_heuristics" / f"{pdf_name}.csv"
        prepared_features_df_path = ai_parsed_dir.parent / "prepared_features" / f"{pdf_name}.csv"

        ai_results_no_heuristics_df_path.parent.mkdir(parents=True, exist_ok=True)
        ai_results_not_combined_df_path = ai_parsed_dir.parent / "ai_parsed_pdf_not_combined" / f"{pdf_name}.csv"
        prepared_features_df_path.parent.mkdir(parents=True, exist_ok=True)
        ai_results_not_combined_df_path.parent.mkdir(parents=True, exist_ok=True)

        # If the files already exist, skip the extraction and feature computation
        # if not overwrite and all(
        #     [
        #         extracted_pdf_path.exists(),
        #         feature_df_path.exists(),
        #         xml_labelled_df_path.exists(),
        #         final_labeled_df_path.exists(),
        #         ai_parsed_df_path.exists(),
        #     ]
        # ):
        #     logger.info(f"Skipping extraction and feature computation for {pdf_name} because the files already exist")
        #     return

        # Step 1: Extract PDF
        print(f"Extracting PDF: {pdf_path.stem}")
        if not extracted_pdf_path.exists() or overwrite:
            extracted_df = extract_pdf_text(pdf_path, pdf_type)
            extracted_df.to_csv(extracted_pdf_path, index=False)
        else:
            extracted_df = pd.read_csv(extracted_pdf_path)

        # Step 2: Compute features
        print(f"Computing features: {pdf_name}")
        if not feature_df_path.exists() or overwrite:
            features_df = compute_features(extracted_df)
            features_df.to_csv(feature_df_path, index=False)

            labelled_pdf_path = labelled_pdf_dir / f"{pdf_name}.csv"
            if labelled_pdf_path.exists():
                update_labelled_pdf_features(feature_df_path, labelled_pdf_path)

        else:
            features_df = pd.read_csv(feature_df_path)
            features_df['pdf_idx'] = features_df.index
            features_df.to_csv(feature_df_path, index=False)

        # Step 3: Parse XML
        print(f"Parsing XML: {xml_path.stem}")
        if not xml_labelled_df_path.exists() or overwrite:
            xml_labelled_df = parsing_method(xml_path)
            xml_labelled_df.to_csv(xml_labelled_df_path, index=False)
        else:
            xml_labelled_df = pd.read_csv(xml_labelled_df_path)

        # # Step 4: Apply labeling
        print(f"Applying labeling: {pdf_name}")
        if not final_labeled_df_path.exists() or overwrite:
            final_labeled_df = sliding_window_labeling(features_df, xml_labelled_df)
            final_labeled_df.to_csv(final_labeled_df_path, index=False)
        else:
            final_labeled_df = pd.read_csv(final_labeled_df_path)

        # Step 5: Parse PDF with AI
        # print(f"Parsing PDF with AI: {pdf_name}")
        model_path = (
            r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\models\RandomForestClassifier.sav"
        )
        if not ai_parsed_df_path.exists() or overwrite or True:
            model = load_model(model_path)
            prepared_features, ai_parsed_df, ai_results_no_heuristics, ai_results_not_combined = parse_pdf_ai(
                pdf_path, model, computed_features_df=features_df, extracted_df=extracted_df
            )
            ai_parsed_df.to_csv(ai_parsed_df_path, index=False)
            ai_results_no_heuristics.to_csv(ai_results_no_heuristics_df_path, index=False)
            ai_results_not_combined.to_csv(ai_results_not_combined_df_path, index=False)
            prepared_features.to_csv(prepared_features_df_path, index=False)
        else:
            ai_parsed_df = pd.read_csv(ai_parsed_df_path)

        logger.info("Pipeline completed successfully!")
        logger.info(f"Final labeled data saved to: {final_labeled_df_path}")

        # Print summary
        print("\n" + "=" * 50)
        print("PIPELINE SUMMARY")
        print("=" * 50)
        print(f"PDF extracted: {extracted_pdf_path}")
        print(f"Features computed: {feature_df_path}")
        print(f"XML labeled: {xml_labelled_df_path}")
        print(f"Final labeled PDF: {final_labeled_df_path}")
        # num_labeled = len(final_labeled_df)
        # num_labeled_above_0_3 = len(final_labeled_df[final_labeled_df['Match_Confidence'] > 0.3])
        # print(f"Percentage Labeled above 0.3 Confidence: {num_labeled_above_0_3 / num_labeled * 100}%")
        print("=" * 50)

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


def main_cfr_xml(overwrite: bool = False) -> None:
    pass

    document_dir = Path(__file__).parents[3] / "data" / "documents" / "CFR"
    # document_dir = Path(r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\validation")
    pdfs = list(document_dir.glob("*.pdf"))
    # pdfs = [CFR_PDF]
    # pdfs = [Path("/home/rmenon/source/ai-pdf-parser/data/documents/CFR/CFR-2024-title21-vol8-chapI-subchapH.pdf")]
    for pdf_path in pdfs:
        xml_path = pdf_path.with_suffix(".xml")
        pdf_type = "CFR"  # Options: "CFR", "FAA_Advisory_Circulars_Data", "Arxiv"
        extract_and_label(pdf_path, xml_path, pdf_type, document_dir, cfr_extracting, overwrite)


def main_xml_easa(overwrite: bool = False) -> None:
    pass

    document_dir = Path(__file__).parents[3] / "data" / "documents" / "EASA"
    pdfs = list(document_dir.glob("*.pdf"))
    pdfs = [
        EASA_DIR / f"{pdf_path}.pdf"
        for pdf_path in [
            'Easy Access Rules for ATM_ANS Equipment _Regulations _EU_ 2023_1769 _ _EU_ 2023_1768_ _PDF_',
            'Easy Access Rules for small category VCA _PDF_',
            'Easy Access Rules for Large Aeroplanes _CS 25_ _Amendment 27_ _PDF_',
            'Easy Access Rules for Master Minimum Equipment List _CS_MMEL_ _Issue 3_ _PDF_',
            'Easy Access Rules for Airborne Communications_ Navigation and Surveillance _CS_ACNS_ Issue 4 _pdf_',
            'Easy Access Rules for Large Rotorcraft _CS_29_ _Amendment 11_ _PDF_',
            'Easy Access Rules for Normal_Category Aeroplanes _CS_23_ _CS Amendment 6_ AMC_GM Issue 4_ _PDF_',
            'Easy Access Rules for Small Rotorcraft _CS_27_ Amendment 10 _pdf_',
            'Easy Access Rules for U_space _PDF_',
            'Easy Access Rules for Aerodromes _PDF_',
            'Easy Access Rules for Information Security _PDF_',
            'Easy Access Rules for Aircrew _Regulation _EU_ No 1178_2011_ _PDF_',
        ]
    ]
    for pdf_path in pdfs:
        xml_path = pdf_path.with_suffix(".xml")
        pdf_type = "EASA"  # Options: "CFR", "FAA_Advisory_Circulars_Data", "Arxiv"
        extract_and_label(pdf_path, xml_path, pdf_type, document_dir, extract_easa_xml, overwrite)


def main_latex(overwrite: bool = False) -> None:
    from ai_doc_parser import DATA_DIR

    logging.basicConfig(level=logging.DEBUG)

    document_dir = DATA_DIR / "documents" / "Latex"
    pdfs = list(document_dir.glob("*.pdf"))
    latex_files = list(document_dir.glob("*.tex"))

    pdfs = [p for p in pdfs]
    print(f"{pdfs=}")

    parser = LatexParser()
    for pdf_path in pdfs:
        pdf_type = "Arvix"  # Options: "CFR",  "FAA_Advisory_Circulars_Data", "Arxiv"
        extract_and_label(pdf_path, latex_files[0], pdf_type, document_dir, parser.parse, overwrite)

    apply_heuristic_to_dir(document_dir / "labelled_pdf", latex_heuristics)


def latex_bullet_list(overwrite: bool = False) -> None:
    from ai_doc_parser import DATA_DIR

    document_dir = DATA_DIR / "documents" / "Bullets"
    pdfs = list(document_dir.glob("*.pdf"))
    latex_files = list(document_dir.glob("*.tex"))

    parser = LatexParser()
    for pdf_path in pdfs:
        pdf_type = "Arvix"  # Options: "CFR",  "FAA_Advisory_Circulars_Data", "Arxiv"
        extract_and_label(pdf_path, latex_files[0], pdf_type, document_dir, parser.parse, overwrite)

        DATA_DIR / "documents"
    apply_heuristic_to_dir(document_dir / "labelled_pdf", latex_heuristics)


def aus_mdr(overwrite: bool = False) -> None:
    from ai_doc_parser import DATA_DIR

    document_dir = DATA_DIR / "documents" / "AUSMDR"
    pdfs = list(document_dir.glob("*.pdf"))
    for pdf_path in pdfs:

        pdf_type = "AUS"
        docx_path = pdf_path.with_suffix(".docx")
        extract_and_label(pdf_path, docx_path, pdf_type, document_dir, extract_docx, overwrite)


def validation(overwrite: bool = False) -> None:
    from ai_doc_parser import DATA_DIR

    doc_dir = DATA_DIR / "documents" / "validation"
    pdf_paths = []
    xml_paths = []
    pdf_types = []

    pdf_paths.append(doc_dir / "CFR-2023-title14-vol5.pdf")
    xml_paths.append(doc_dir / "CFR-2023-title14-vol5.xml")
    pdf_types.append("CFR")

    latex_names = ['v19', 'v51', 'v60', 'v65', 'v70', 'v79', 'v82', 'v86', 'v89', 'v91', 'v94', 'v97']

    for latex_name in latex_names:
        pdf_paths.append(doc_dir / f"{latex_name}.pdf")
        xml_paths.append(doc_dir / f"inspird.tex")
        pdf_types.append("Latex")
    # Add Latex Bullets
    pdf_paths.append(doc_dir / "B2LILNN15.pdf")
    xml_paths.append(doc_dir / "inspird_bullets.tex")
    pdf_types.append("Latex")

    # Append AUSMDR
    pdf_paths.append(doc_dir / "devices-argmd-01.pdf")
    xml_paths.append(doc_dir / "devices-argmd-01.docx")
    pdf_types.append("AUS")

    parser = LatexParser()
    for pdf_path, xml_path, pdf_type in zip(pdf_paths, xml_paths, pdf_types):
        if pdf_type == "CFR":
            parsing_method = cfr_extracting
        elif pdf_type == "Latex":
            parsing_method = parser.parse
        elif pdf_type == "AUS":
            parsing_method = extract_docx
        else:
            raise ValueError(f"Invalid PDF type: {pdf_type}")
        extract_and_label(pdf_path, xml_path, pdf_type, doc_dir, parsing_method, overwrite)


def nist(overwrite: bool = False) -> None:
    from ai_doc_parser import DATA_DIR

    document_dir = DATA_DIR / "documents" / "NIST"
    pdfs = list(document_dir.glob("*.pdf"))
    for pdf_path in pdfs:
        pdf_type = "NIST"
        extract_and_label(pdf_path, pdf_path, pdf_type, document_dir, extract_docx, overwrite)


if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    overwrite = False
    # main_cfr_xml(overwrite=overwrite)
    # main_xml_easa(overwrite=overwrite)
    # main_latex(overwrite=overwrite)
    # latex_bullet_list(overwrite=overwrite)
    # aus_mdr(overwrite=overwrite)
    # nist(overwrite=overwrite)
    # validation(overwrite=overwrite)

    document_dir = DATA_DIR / "documents"

    # apply_heuristic_to_dir(document_dir / "Latex" / "labelled_pdf", latex_heuristics)
    # apply_heuristic_to_dir(document_dir / "Bullets" / "labelled_pdf", latex_heuristics)
