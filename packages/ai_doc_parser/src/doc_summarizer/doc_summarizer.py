#!/usr/bin/env python3
"""
PDF Text Parser and Summarizer using AIPDFParser_V5_rmenon
This script demonstrates how to use the AIPDFParser_V5_rmenon module to parse PDF files into structured text
and then summarize them using BERT-based extractive and abstractive summarization.
"""

import json
import logging
import os
import re
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from database.document_service import (
    find_document_by_id,
    get_all_document_ids,
    get_paragraph_vectors_to_be_indexed,
    set_document_summary,
)
from database.entity.ScriptsProperty import ScriptsConfig, parseCredentialFile

# Import the PDF parser module
from database.utils.MySQLFactory import MySQLDriver

# BERT and NLP imports


os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def format_text_to_markdown(markdown_text: str) -> str:
    """
    Format text into proper markdown with clean structure.
    """
    markdown_text = markdown_text.replace("- **", "\n- **")
    markdown_text = markdown_text.replace(" #", "\n#")
    return markdown_text


def is_content(text: str, pred_type: int) -> bool:
    """Determine if a text block is content."""
    if pred_type == 0:  # Paragraph
        return True
    return False


def is_toc(text: str, pred_type: int) -> bool:
    """Determine if a text block is a table of contents."""
    if pred_type == 4:  # Table of Contents
        return True
    return False


def is_heading(text: str, pred_type: int) -> bool:
    """Determine if a text block is a heading."""
    # Check prediction type first
    # Additional heuristics for heading detection
    text = text.strip()
    if not text:
        return False

    if pred_type == 2:  # Heading
        return True

    return False


def is_subheading(text: str, pred_type: int) -> bool:
    """Determine if a text block is a subheading."""
    if pred_type == 3:  # Heading continued line
        return True

    text = text.strip()
    if not text:
        return False

    # Check for subheading patterns
    subheading_patterns = [
        r"^\d+\.\d+\.\d+\s+[A-Z]",  # 1.1.1. Subsubtitle
        r"^[a-z]\.\s+[A-Z]",  # a. Subtitle
        r"^\([a-z]\)\s+[A-Z]",  # (a) Subtitle
    ]

    for pattern in subheading_patterns:
        if re.match(pattern, text):
            return True

    return False


def split_text_into_chunks(text: str, max_words: int) -> List[str]:
    """Split text into chunks of maximum word count."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)

    return chunks


@dataclass
class SectionChunk:
    # Content is section heading:content
    content: Dict[str, str] = field(default_factory=dict)
    page: int = 0
    word_count: int = 0
    chunk_id: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "SectionChunk":
        return cls(
            content=data["content"],
            page=data["page"],
            word_count=data["word_count"],
            chunk_id=data["chunk_id"],
        )


@dataclass
class SectionSummary:
    chunk: SectionChunk
    summary: str
    prompt: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "SectionSummary":
        return cls(
            chunk=SectionChunk.from_dict(data["chunk"]),
            summary=data["summary"],
            prompt=data["prompt"],
        )


class DocumentSummarizer:
    """
    A comprehensive document summarizer that uses BERT for extractive summarization
    and GenAI for summarization.
    """

    def __init__(
        self,
        llm_endpoint="http://localhost:11434/api/chat",
        max_chunk_size: int = 500,
        max_llm_calls: int = 50,
        max_paragraph_words: int = 128,
    ):
        """
        Initialize the document summarizer.

        Args:
            bert_model_name: Name of the BERT model to use for embeddings
            max_chunk_size: Maximum words per section chunk
            max_paragraph_words: Maximum words per paragraph for extractive summary
        """
        self.llm_endpoint = llm_endpoint
        # Number of words per chunk
        self.max_chunk_size = max_chunk_size
        self.max_llm_calls = max_llm_calls
        self.max_paragraph_words = max_paragraph_words
        self.section_chunks = []

    def organize_into_sections(
        self, parsed_content: List[List[Tuple[str, int, int]]]
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Organize parsed content into sections based on headings.

        Args:
            parsed_content: Output from parse_pdf_to_text

        Returns:
            List of section dictionaries with heading and content
        """
        sections = []
        current_heading = None
        current_content = []

        def has_keyword(text: str, keywords: List[str]) -> bool:
            """Check if the text contains any of the keywords."""
            return any(keyword in text.lower() for keyword in keywords)

        parsed_content = [p[0] for p in parsed_content]

        # If there is a summary section, use it, otherwise use the sections that are not preface sections
        summary_keywords = ["executive summary", "abstract", "key takeaways"]

        summary_sections = []

        def append_section():
            nonlocal current_heading, current_content, sections, summary_sections
            content = "\n".join(current_content)
            section = {
                "heading": current_heading,
                "content": content,
                "page": page_num,
            }
            sections.append(section)
            if has_keyword(section["heading"], summary_keywords):
                summary_sections.append(section)

        num_pages = max(page_num for _, _, page_num in parsed_content)
        for text, pred_type, page_num in parsed_content:
            text = text.strip()
            if not text:
                continue

            # Check if this is a heading
            if is_heading(text, pred_type):
                # Save previous section if exists
                if current_heading and current_content:
                    current_heading = f"{current_heading}"
                    append_section()
                    # Start new section
                    current_heading = text
                    current_content = []
                elif current_heading and not current_content:
                    current_heading = f"{current_heading} - {text}"
                else:
                    # Start new section
                    current_heading = text
                    current_content = []

            elif is_toc(text, pred_type):
                if current_heading and current_content:
                    append_section()
                # if the page number is greater than 10% of the total pages, start a new section
                if page_num > 0.1 * num_pages:
                    sections = []
                    current_heading = None
                    current_content = []

            # Check if this is a subheading (continue with current section)
            elif is_subheading(text, pred_type):
                # Add subheading to current content
                if current_content:
                    current_content.append(f"\n**{text}**")
                else:
                    current_content.append(text)

            # Regular content
            elif is_content(text, pred_type):
                current_heading = current_heading or "Introduction"
                current_content.append(text)
            else:
                # Other types of text like HEADER, FOOTER, TOC, TABLE are ignored
                continue

        # Add the last section
        if current_heading and current_content:
            append_section()

        if summary_sections:
            log.debug("Found %d summary sections", len(summary_sections))
            log.debug(
                "Summary sections: %s",
                [section["heading"] for section in summary_sections],
            )
            return summary_sections, True

        # remove sections that have headers with the keyword "definitions" in it
        def is_valid_section(section: Dict[str, Any]) -> bool:
            bad_heading_keywords = [
                "definition",
                "abbreviation",
                "symbols",
                "terms",
                "glossary",
                "acronyms",
            ]
            for keyword in bad_heading_keywords:
                if keyword in section["heading"].lower():
                    return False

            if len(section["content"]) < len(section["heading"]):
                return False
            return True

        sections = [section for section in sections if is_valid_section(section)]
        return sections, False

    def create_section_chunks(
        self,
        sections: List[Dict[str, Any]],
        min_words_per_chunk: int = 10,
        max_response_length: int = 5000,
    ) -> List[SectionChunk]:
        """
        Create section chunks, subdividing if they exceed max_chunk_size.

        Args:
            sections: List of section dictionaries

        Returns:
            List of section chunks
        """

        total_word_count = sum(len(section["content"].split()) for section in sections)
        max_chunk_size = max(
            self.max_chunk_size * 3, total_word_count // self.max_llm_calls
        )

        current_chunk: SectionChunk = SectionChunk()
        section_chunks: List[SectionChunk] = []

        for section in sections:
            heading = section["heading"]
            content = section["content"]
            page = section["page"]

            # Combine all content into one text
            word_count = len(content.split())
            if current_chunk.word_count + word_count > max_chunk_size:
                section_chunks.append(current_chunk)
                current_chunk = SectionChunk(
                    content={}, page=0, word_count=0, chunk_id=0
                )
            current_chunk.content[heading] = content
            current_chunk.word_count += word_count
            current_chunk.page = page
            current_chunk.chunk_id = len(section_chunks)

        if current_chunk.word_count > 0:
            section_chunks.append(current_chunk)

        # filter out chunks with less than min_words_per_chunk
        section_chunks = [
            chunk for chunk in section_chunks if chunk.word_count > min_words_per_chunk
        ]
        return section_chunks

    def ollama_generate(self, prompt: dict) -> str:
        """Call local LLM (Ollama) to generate a summary."""
        model = "llama3.2:latest"

        context_length = len(str(prompt).split(" ")) * 3
        log.debug("Generating with context length: %d", context_length)
        r = requests.post(
            self.llm_endpoint,
            timeout=120,
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"{prompt}",
                    }
                ],
                "options": {
                    "top_k": 5,
                    "temperature": 0.0,
                    "num_ctx": context_length,
                },
            },
            stream=True,
        )
        r.raise_for_status()
        #  Safety limit for content length
        max_line_count = len(str(prompt).split(" "))
        print(f"Max line count: {max_line_count}")

        # Use list to collect response parts for efficient string building
        response_parts = []
        line_count = 0

        for line in r.iter_lines():
            line_count += 1

            if line_count > max_line_count:
                log.warning(
                    f"Reached maximum line count ({max_line_count}), breaking loop"
                )
                break

            body = json.loads(line)
            # Check for errors first (early exit)
            if "error" in body:
                raise Exception(body["error"])

            # Check for completion early
            if body.get("done", False):
                log.debug(f"Stream completed normally after {line_count} lines")
                break

            # Extract content more efficiently
            response_part = body.get("message", "")
            if isinstance(response_part, dict):
                content = response_part.get("content", "")
            elif isinstance(response_part, str):
                content = response_part
            else:
                content = ""

            if content:  # Only append non-empty content
                response_parts.append(content)

        # Join all parts at once (much more efficient than string concatenation)
        summary_text = "".join(response_parts)

        log.info(
            "Generated summary: %d characters from %d lines",
            len(summary_text),
            line_count,
        )
        return summary_text

    def summarize_chunk(
        self,
        chunk: SectionChunk,
    ) -> SectionSummary:
        """
        Summarization using local LLM (Ollama).
        """
        prompt = {
            "context": chunk.content,
            "system prompt": (
                "You are an expert technical writer. Your task is to produce a replacement summary of the provided text. "
                "The summary will be combined with other summaries to create a final summary.\n"
                "The summary must:\n"
                "1. Stand alone as if it were the original text, not a description of it.\n"
                "2. Contain all key information in a concise, comprehensive, and technically accurate way.\n"
                "3. Preserve domain-specific terminology.\n"
                "4. Be written in clear, direct sentences that present the information itself.\n"
                "5. Write the summary and no other text.\n"
            ),
            "user prompt": (
                "Rewrite the provided 'context' into a concise, standalone summary that fully replaces it.\n"
                "Use only the information in 'context'.\n"
            ),
        }
        response = self.ollama_generate(prompt)
        return SectionSummary(chunk, response, prompt)

    def create_final_summary(
        self, section_summaries: List[SectionSummary]
    ) -> Tuple[dict, str]:
        """
        Create a final 1-page summary combining all section summaries using local LLM.
        """
        summaries = [section.summary for section in section_summaries]
        try:
            prompt = {
                # Put ONLY the already-made summaries here; do NOT include rules/goals in this string.
                "summaries_block": summaries,
                "system prompt": (
                    "ROLE: Technical editor.\n"
                    "TASK: Produce a single, standalone executive summary that condenses the provided summaries into a single page. Only give the summary and no other text.\n"
                ),
                "user prompt": (
                    "From 'summaries_block', write a Markdown executive summary."
                    "Output only the summary in Markdown—no preamble, no commentary."
                ),
            }

            raw_summary = self.ollama_generate(prompt)

            # Format the summary using the markdown formatter
            formatted_summary = format_text_to_markdown(raw_summary)
            return prompt, formatted_summary
        except Exception as e:
            log.debug("Error creating final summary: %s", e)
            return {}, "Error creating final summary"

    def summarize_document(
        self,
        parsed_content: List[List[Tuple[str, int, int]]],
        save_summaries: bool = True,
        doc_name: str = "",
        compute_section_summaries: bool = True,
        min_words_per_chunk: int = 10,
    ) -> Dict[str, Any]:
        """
        Main method to summarize a parsed document.

        Args:
            parsed_content: Output from parse_pdf_to_text

        Returns:
            Dictionary containing all summaries
        """

        def save_json(data: dict, file_path: Path | str):
            # if file path does not have a parent directory, set it to output_dir
            if isinstance(file_path, str):
                file_path = Path(file_path)
            if file_path.parent == Path("."):
                file_path = output_dir / file_path
            if save_summaries:
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=2)

        if save_summaries and not doc_name:
            raise ValueError("doc_name is required when save_summaries is True")
        output_dir = Path("./document_summaries") / doc_name

        if save_summaries:
            output_dir.mkdir(parents=True, exist_ok=True)
            log.debug("Saving summaries to %s", output_dir.absolute())

        log.debug("Starting document summarization...")

        # Organize into sections
        log.debug("Organizing content into sections...")
        sections, is_executive_summary = self.organize_into_sections(parsed_content)
        save_json(sections, f"{doc_name}_sections.json")

        log.debug("Found %d sections", len(sections))

        # Create section chunks
        log.debug("Creating section chunks...")
        section_chunks = self.create_section_chunks(sections, min_words_per_chunk)
        section_chunk_dicts = [asdict(chunk) for chunk in section_chunks]
        save_json(section_chunk_dicts, f"{doc_name}_section_chunks.json")
        log.debug("Created %d section chunks", len(section_chunks))

        section_summaries: List[SectionSummary]
        if not compute_section_summaries:
            with open(output_dir / f"{doc_name}_section_summaries.json", "r") as f:
                section_summaries = [
                    SectionSummary.from_dict(summary) for summary in json.load(f)
                ]
        elif is_executive_summary:
            section_summaries = []
            for chunk in section_chunks:
                extracted_summary = list(chunk.content.values())[0]
                section_summaries.append(SectionSummary(chunk, chunk.content))
        else:
            section_summaries = []
            for i, chunk in enumerate(section_chunks):
                log.debug("Processing chunk %d/%d", i + 1, len(section_chunks))
                chunk_summary = self.summarize_chunk(chunk)
                section_summaries.append(chunk_summary)
                save_json(
                    [asdict(s) for s in section_summaries],
                    f"{doc_name}_section_summaries.json",
                )

        # Create final summary
        log.debug("Creating final executive summary...")

        if is_executive_summary and len(section_summaries) == 1:
            if isinstance(section_summaries[0].summary, dict):
                final_summary = list(section_summaries[0].summary.values())[0]
            else:
                final_summary = section_summaries[0].summary
            final_summary_prompt = {}
        else:
            # Create final summary from multiple executive summary sections
            final_summary_prompt, final_summary = self.create_final_summary(
                section_summaries
            )
        save_json(final_summary_prompt, f"{doc_name}_final_summary_prompt.json")
        save_json(final_summary, f"{doc_name}_final_summary.json")
        with open(output_dir / f"{doc_name}_final_summary.md", "w") as f:
            f.write(str(final_summary))

        # Compile results
        results = {
            "document_info": {
                "total_sections": len(sections),
                "total_chunks": len(section_chunks),
                "total_words": sum(chunk.word_count for chunk in section_chunks),
                "timestamp": datetime.now().isoformat(),
            },
            "section_summaries": section_summaries,
            "final_summary": final_summary,
            "final_summary_prompt": final_summary_prompt,
        }

        log.debug("Document summarization completed!")
        return results


def has_preface(text: str) -> bool:
    """
    Check if the text contains a preface.
    """
    return not text.startswith("**")


def remove_preface(text: str) -> str:
    """
    Remove the preface from the text.
    """
    if has_preface(text):
        return "** ".join(text.split("**")[1:])
    return text


def main(config: ScriptsConfig, docId: int):
    """
    Example usage of the PDF parser and summarizer, modified to load a single document from SQL database.
    """
    mysql_driver = MySQLDriver(cred=config.databaseConfig.__dict__)

    # Fetch the document by ID

    doc = find_document_by_id(mysql_driver, docId)
    if not doc:
        log.debug("No document found in the database with ID %d.", docId)
        return
    if doc.summary:
        # if has_preface(doc.summary):
        #     summary = remove_preface(doc.summary)
        #     set_document_summary(mysql_driver, doc, summary)
        # else:
        #     print(f"Document already summarized: {doc.documentId} - {getattr(doc, 'title', '')}")
        #     return
        # print(f"Document already summarized: {doc.documentId} - {getattr(doc, 'title', '')}")
        # return
        pass

    log.info("Selected document: %d - %s", doc.documentId, getattr(doc, "title", ""))

    log.info("Document URL: %s", doc.url)

    # Get paragraphs for the selected document
    paragraphs = get_paragraph_vectors_to_be_indexed(mysql_driver, doc)
    if not paragraphs:
        log.debug("No paragraphs found for the selected document.")
        return

    # Convert paragraphs to the expected format for summarization
    parsed_content = []
    for para in paragraphs:
        parsed_content.append(
            [
                (
                    getattr(para, "data", ""),
                    getattr(para, "Type", 0),
                    getattr(para, "Page_No", 0),
                )
            ]
        )

    if parsed_content:
        log.debug("\nParsed %d paragraphs from PDF:", len(parsed_content))
        log.debug("=" * 60)

        summarizer = DocumentSummarizer(
            llm_endpoint="http://localhost:11434/api/chat",
            max_chunk_size=500,
            max_paragraph_words=128,
        )
        # get only alphanumeric characters and replace spaces with underscores
        title = "".join(c for c in doc.title if c.isalnum() or c == " ")
        title = title.replace(" ", "_")[:15]
        doc_name = f"{doc.documentId}_{title}"
        summary_results = summarizer.summarize_document(
            parsed_content,
            save_summaries=True,
            doc_name=doc_name,
            compute_section_summaries=True,
        )
        log.info("EXECUTIVE SUMMARY:\n %s", summary_results["final_summary"])
        # print("\n" + "="*60)
        # print("FINAL EXECUTIVE SUMMARY")
        # print("="*60)
        # print(summary_results['final_summary'])
        # print(f"\nSummary Statistics:")
        # print(f"- Total sections: {summary_results['document_info']['total_sections']}")
        # print(f"- Total chunks: {summary_results['document_info']['total_chunks']}")
        # print(f"- Total words: {summary_results['document_info']['total_words']}")
        # print(f"- Summaries saved to: ./document_summaries/")
        set_document_summary(mysql_driver, doc, summary_results["final_summary"])
    else:
        log.debug("Failed to load or parse document from database.")


if __name__ == "__main__":

    try:
        props = None
        docIdsList = []
        if len(sys.argv) > 1:
            n = len(sys.argv[1])
            docId = int(sys.argv[1])
        else:
            docId = 12

        # configs = parseCredentialFile('/app/tlp_config.json')
        configs = parseCredentialFile(
            # "/dockers/Enginius/test/scripts/rmfdemo-tlp_config.json"
            # "/dockers/Enginius/test/scripts/test-tlp_config.json"
            "/dockers/Enginius/test/scripts/testmed-tlp_config.json"
        )
        if configs:
            mysql_driver = MySQLDriver(cred=configs.databaseConfig.__dict__)
            docIdsList = get_all_document_ids(mysql_driver)
            print(f"{docIdsList=}")
            # run(configs, docIdsList)
            # docs = {
            #     # "EASA Long Document": 784,
            #     # "EASA Short Document": 796,
            #     # "Advisory Circular Long Document": 67,
            #     # "Advisory Circular Short Document": 177,
            #     "CFR": 1,
            #     # "Short Document": 703,
            #     # "Long Document": 791,
            # }
            # docIdsList = [107]
            for docId in docIdsList:
                if docId < 100:
                    continue
                print(
                    f"\n\n----------------------- Summarizing Document: {docId} ------------------------"
                )
                main(configs, docId)
                print()
        # run_multi_threading(configs)
    except Exception as e:
        log.debug(traceback.format_exc())
        traceback.print_exc()
        log.debug("Exception: %s", e)
