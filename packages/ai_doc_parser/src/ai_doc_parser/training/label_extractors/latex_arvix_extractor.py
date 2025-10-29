"""
Clean LaTeX Parser for Academic Documents

This module provides functionality to parse LaTeX documents and extract structured
paragraphs with classification labels. It's designed to process academic papers
and convert them into machine-readable formats.

Author: AI Assistant
Date: 2024
"""

import logging

from ai_doc_parser.training.post_labelling_heuristics import latex_heuristics

log = logging.getLogger(__name__)
import re
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from plasTeX.TeX import TeX

from ai_doc_parser.text_class import TextClass

log = logging.getLogger(__name__)


def make_row(text: str, text_class: TextClass) -> Dict[str, Any]:

    return {
        'text': latex_to_text(text).strip(),
        'PageNumber': -1,
        'SourceClass': text_class.value,
        'SourceClassName': text_class.name,
    }


def latex_to_text(s: str) -> str:
    """
    Convert LaTeX to plain text by removing commands/environments and keeping readable content.
    - Keeps the *contents* of {...} after commands (e.g., \textbf{hi} -> "hi").
    - Removes \begin{...}/\end{...} wrappers but keeps their inner text.
    - Removes optional args [..].
    - Strips math delimiters ($...$, $$...$$, \(...\), \[...\]) but keeps the math text.
    - Unescapes common LaTeX escapes (\%, \_, \&, \$, \{, \}, \\).
    - Removes leftover commands like \alpha, \LaTeX (drops the command entirely).
    - Collapses whitespace.
    NOTE: This is regex-based; it handles common cases but isn't a full TeX parser.
    """

    # 0) Normalize line endings
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # 1) Remove LaTeX comments (unescaped % to end of line)
    s = re.sub(r'(?<!\\)%.*', '', s)

    # 2) Remove \begin{...} and \end{...} wrappers but keep contents
    s = re.sub(r'\\begin\{[^}]*\}', '', s)
    s = re.sub(r'\\end\{[^}]*\}', '', s)

    # 3) Protect escaped \$ so we don’t treat it as math delimiter
    s = s.replace(r'\$', '<<ESCAPED_DOLLAR>>')

    # 4) Strip math delimiters, keep inner text
    # $$...$$ (greedy across lines)
    s = re.sub(r'\$\$(.*?)\$\$', r'\1', s, flags=re.DOTALL)
    # $...$ (avoid crossing $$ regions already handled)
    s = re.sub(r'\$(.*?)\$', r'\1', s, flags=re.DOTALL)
    # \[...\], \(...\)
    s = re.sub(r'\\\[(.*?)\\\]', r'\1', s, flags=re.DOTALL)
    s = re.sub(r'\\\((.*?)\\\)', r'\1', s, flags=re.DOTALL)

    # 5) Drop \verb constructs, keep inner text (simple forms)
    s = re.sub(r'\\verb\*?(.)(.*?)\1', r'\2', s, flags=re.DOTALL)

    # 5.5) Remove citation commands entirely (\cite, \citet, \citep, etc.) with all args
    #      Replace with a single space to avoid concatenating words
    s = re.sub(r'\\cite[a-zA-Z*]*\s*(?:\[[^\]]*\]\s*)*(?:\{[^{}]*\}\s*)+', ' ', s)

    # 6) Iteratively replace commands with braced args: \cmd{A}{B}[opt] -> "A B"
    #    This keeps the human text inside braces and discards the command + options.
    cmd_with_args = re.compile(
        r'''
        \\[a-zA-Z]+[*]?      # command name (with optional *)
        (?:\s*\[[^\]]*\])*\s* # optional [..] args (any number)
        (?:\{[^{}]*\})+       # one or more {...} groups
        ''',
        re.VERBOSE | re.DOTALL,
    )

    def _keep_brace_text(m):
        return ' '.join(re.findall(r'\{([^{}]*)\}', m.group(0)))

    while True:
        new_s = cmd_with_args.sub(_keep_brace_text, s)
        if new_s == s:
            break
        s = new_s

    # 7) Remove remaining standalone optional args after commands (rare stragglers)
    s = re.sub(r'\\[a-zA-Z]+[*]?\s*\[[^\]]*\]', '', s)

    # 8) Remove remaining simple commands without args like \LaTeX, \alpha, \emph (bare)
    s = re.sub(r'\\[a-zA-Z]+[*]?', '', s)

    # 9) Unescape common LaTeX escapes and symbols
    replacements = {
        r'\%': '%',
        r'\_': '_',
        r'\&': '&',
        r'\#': '#',
        r'\{': '{',
        r'\}': '}',
        r'\\': '\\',
        '``': '“',
        "''": '”',
        '---': '—',  # em-dash
        '--': '–',  # en-dash
        r'\~{}': '~',
        r'\^{}': '^',
        '~': ' ',  # non-breaking space -> regular space
    }
    # Restore escaped dollars
    s = s.replace('<<ESCAPED_DOLLAR>>', '$')
    for k, v in replacements.items():
        s = s.replace(k, v)

    # 10) Remove leftover braces that are just grouping, but keep content already extracted
    #     (Be conservative: only strip if they don't look like part of code)
    s = s.replace('{', '').replace('}', '')

    # 11) Collapse whitespace
    s = re.sub(r'[ \t\f\v]+', ' ', s)
    s = re.sub(r'\n\s*\n+', '\n\n', s)  # collapse blank lines
    s = s.strip()

    return s


class LatexParser:
    """
    A clean and maintainable LaTeX document parser.

    This class provides methods to parse LaTeX files, extract paragraphs,
    and classify them with various labels (headers, tables, etc.).
    """

    # LaTeX commands that indicate headers or special content
    HEADER_COMMANDS = [
        "\\subsection",
        "\\subsubsection",
        "\\section",
        "\\title",
    ]

    # Commands that indicate table environments
    TABLE_COMMANDS = [
        "\\begin{table",
        "\\end{table",
    ]

    # Patterns for paragraph breaks
    PARAGRAPH_BREAK_PATTERNS = [
        r"(\n+\s*%?\s*\n+)",  # Multiple newlines
        r"(\\par)",  # Paragraph command
        r"(\\indent)",  # Indent command
        r"(\\noindent)",  # No indent command
        r"(\\\\\\)",  # Double backslash
        r"(\\item)",  # List item
    ]

    def __init__(self):
        """Initialize the LaTeX parser."""
        self.header_mapping: Dict[int, str] = {}

    def _is_table_of_contents(self, line: str) -> bool:
        """
        Check if a line contains a table of contents command.

        Args:
            line: The line to check

        Returns:
            True if line contains TOC command, False otherwise
        """
        return "\\tableofcontents" in line

    def _extract_headers(self, text_lines: List[str]) -> Dict[int, str]:
        """
        Extract header information from LaTeX text.

        Args:
            text_lines: List of text lines from the LaTeX file

        Returns:
            Dictionary mapping line numbers to header text
        """
        header_mapping = OrderedDict()
        current_header = ""
        in_header = False
        line_number = 0

        for line in text_lines:
            line_number += 1

            # Check if line starts with a header command
            is_header_start = any(line.strip().startswith(cmd) for cmd in self.HEADER_COMMANDS)

            if is_header_start:
                if "}" in line:
                    # Single-line header
                    header_mapping[line_number] = line
                else:
                    # Multi-line header
                    current_header = line
                    in_header = True
            elif in_header:
                current_header += line
                if "}" in line:
                    # End of multi-line header
                    header_mapping[line_number] = current_header
                    current_header = ""
                    in_header = False

        return header_mapping

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split LaTeX text into paragraphs using various break patterns.

        Args:
            text: The LaTeX text to split

        Returns:
            List of paragraph strings
        """
        # Combine all paragraph break patterns
        break_pattern = "|".join(self.PARAGRAPH_BREAK_PATTERNS)

        # Split text and filter out empty paragraphs
        paragraphs = [par.strip() for par in re.split(break_pattern, text) if par and par.strip()]

        return paragraphs

    def _process_latex_environments(self, paragraphs: List[str]) -> List[str]:
        """
        Process LaTeX environments (like tables) by splitting around begin/end commands.

        Args:
            paragraphs: List of paragraphs to process

        Returns:
            List of processed paragraphs
        """
        processed_paragraphs = []

        for paragraph in paragraphs:
            lines = paragraph.split("\n")
            split_paragraphs = []

            # Handle \begin{environment} commands
            if "begin{" in paragraph:
                begin_indices = [i for i, line in enumerate(lines) if "begin{" in line]

                for idx in reversed(begin_indices):
                    if idx + 1 >= len(lines) or lines[idx + 1].startswith("\\"):
                        continue

                    # Split around the \begin{ command
                    before = "\n".join(lines[:idx])
                    after = "\n".join(lines[idx + 1 :])
                    split_paragraphs = [before, after] if before.strip() else [after]

            # Handle \end{environment} commands
            if "end{" in paragraph:
                if split_paragraphs:
                    # Process existing splits
                    new_splits = []
                    for split_par in split_paragraphs:
                        par_lines = split_par.split("\n")
                        end_indices = [i for i, line in enumerate(par_lines) if "end{" in line]

                        for idx in reversed(end_indices):
                            if idx - 1 < 0 or par_lines[idx - 1].startswith("\\"):
                                new_splits.append(split_par)
                                continue

                            # Split around the \end{ command
                            before = "\n".join(par_lines[:idx])
                            after = "\n".join(par_lines[idx + 1 :])
                            new_splits.extend([before, after] if before.strip() else [after])

                    split_paragraphs = new_splits
                else:
                    # Handle \end{ commands in original paragraph
                    end_indices = [i for i, line in enumerate(lines) if "end{" in line]

                    for idx in reversed(end_indices):
                        if idx - 1 < 0 or lines[idx - 1].startswith("\\"):
                            continue

                        before = "\n".join(lines[:idx])
                        after = "\n".join(lines[idx + 1 :])
                        split_paragraphs = [before, after] if before.strip() else [after]

            # Add processed paragraphs
            if split_paragraphs:
                processed_paragraphs.extend(split_paragraphs)
            else:
                processed_paragraphs.append(paragraph)

        return processed_paragraphs

    def _remove_section_commands(self, paragraphs: List[str]) -> List[str]:
        """
        Remove \\section commands from paragraph content while preserving the content.

        Args:
            paragraphs: List of paragraphs to process

        Returns:
            List of paragraphs with section commands removed
        """
        processed_paragraphs = []

        for paragraph in paragraphs:
            lines = paragraph.split("\n")

            # Find and remove \section lines
            section_indices = [i for i, line in enumerate(lines) if "\\section" in line]

            for idx in reversed(section_indices):
                if idx + 1 < len(lines) and not lines[idx + 1].startswith("\\"):
                    del lines[idx]

            processed_paragraph = "\n".join(lines)
            if processed_paragraph.strip():
                processed_paragraphs.append(processed_paragraph)

        return processed_paragraphs

    def _clean_latex_commands(self, paragraphs: List[str]) -> List[str]:
        """
        Remove LaTeX commands, comments, and other markup from paragraphs.

        Args:
            paragraphs: List of paragraphs to clean

        Returns:
            List of cleaned paragraphs
        """
        cleaned_paragraphs = []

        for paragraph in paragraphs:
            lines = paragraph.split("\n")

            # Remove \label commands
            lines = [line for line in lines if not line.strip().startswith("\\label")]

            # Remove comment lines (starting with %)
            lines = [line for line in lines if not line.strip().startswith("%")]

            # Remove other LaTeX commands (but keep headers and author info)
            filtered_lines = []
            for line in lines:
                if (
                    line.strip().startswith("\\")
                    and not line.strip().startswith("\\author")
                    and not any(line.strip().startswith(cmd) for cmd in self.HEADER_COMMANDS)
                ):
                    continue
                filtered_lines.append(line)

            cleaned_paragraph = "\n".join(filtered_lines)
            if cleaned_paragraph.strip():
                cleaned_paragraphs.append(cleaned_paragraph)

        return cleaned_paragraphs

    def _normalize_text(self, paragraph: str) -> str:
        """
        Normalize paragraph text by removing LaTeX commands and symbols.

        Args:
            paragraph: Paragraph text to normalize

        Returns:
            Normalized paragraph text
        """
        # Basic LaTeX command patterns to remove
        latex_patterns = [
            r'\\[a-zA-Z]+(\{[^}]*\})?',  # LaTeX commands with optional arguments
            r'\\[a-zA-Z]+',  # Simple LaTeX commands
            r'\\[^\w\s]',  # LaTeX symbols
        ]

        # Remove LaTeX commands and symbols
        normalized = paragraph
        for pattern in latex_patterns:
            normalized = re.sub(pattern, '', normalized)

        # Handle common LaTeX symbols
        replacements = {
            '\\&': '&',
            '\\%': '%',
            '\\$': '$',
            '\\#': '#',
            '\\_': '_',
            '\\{': '{',
            '\\}': '}',
            '\\~': '~',
            '\\^': '^',
            '\\"': '"',
            "\\'": "'",
        }

        for latex_symbol, replacement in replacements.items():
            normalized = normalized.replace(latex_symbol, replacement)

        # Remove extra whitespace and normalize
        normalized = re.sub(r'\s+', ' ', normalized).strip().lower()

        return normalized

    def _classify_paragraph(self, paragraph: str, last_header_match: int) -> Tuple[int, int, int]:
        """
        Classify a single paragraph as header, table, or regular text.

        Args:
            paragraph: The paragraph to classify
            last_header_match: Index of last header match for tracking

        Returns:
            Tuple of (is_header, is_table, is_paragraph, new_last_header_match)
        """
        # Check if paragraph is a header
        is_header = False
        new_last_header_match = last_header_match

        for i, header_text in enumerate(self.header_mapping.values()):
            if header_text == paragraph and i >= last_header_match + 1:
                is_header = True
                new_last_header_match = i
                break

        # Check if paragraph is part of a table
        is_table = "\\begin{table" in paragraph or "\\end{table" in paragraph

        # Regular paragraphs are not headers
        is_paragraph = not is_header

        return is_header, is_table, is_paragraph, new_last_header_match

    def classify_latex_block(
        self,
        block: Any,
        parent_class: Optional[TextClass] = None,
        is_first_child: bool = False,
    ) -> List[Dict[str, Any]]:
        unsed_node_names = [
            "documentclass",
            "#text",
            "usepackage",
            "section",
            "paragraph",
            "item",
            "enumerate",
            "itemize",
        ]
        node_name = block.nodeName
        first_child = block.childNodes[0] if len(block.childNodes) > 0 else None

        if first_child and first_child.nodeName in ["itemize", "enumerate"]:
            return self.classify_latex_block(first_child, TextClass.BULLET_LIST)
        elif first_child and first_child.nodeName == "abstract":
            rows = [make_row("Abstract", TextClass.HEADING)]
            text = " ".join([child.source for child in first_child._dom_childNodes if child.source != ""])
            rows.append(make_row(text, TextClass.PARAGRAPH))
            return rows
        elif node_name == "itemize":
            return self._process_child_nodes(block.childNodes, TextClass.BULLET_LIST)
        elif node_name == "enumerate":
            return self._process_child_nodes(block.childNodes, TextClass.ENUM_LIST)
        elif node_name == "par":
            text = latex_to_text(block.source)
            if not text.strip():
                return []
            return [make_row(text, TextClass.PARAGRAPH)]
        elif node_name == "item":
            item_text = block.childNodes[0].source
            return [make_row(item_text, parent_class)]
        elif node_name == "title":
            title_name = " ".join([child.source for child in block.childNodes[0]._dom_childNodes])
            return [make_row(title_name, TextClass.HEADING)]
        elif node_name == "document":
            return self._process_child_nodes(block.childNodes)
        elif node_name.replace("sub", "") == "section":
            section_name = block.attributes['title'].source
            header_row = [make_row(section_name, TextClass.HEADING)]
            child_rows = self._process_child_nodes(block.childNodes)
            return header_row + child_rows
        elif node_name == "paragraph":
            return [make_row(block.source, TextClass.PARAGRAPH)]
        elif node_name in unsed_node_names:
            return []
        else:
            log.warning(f"Unused node name: {node_name}")
            return []

    def _process_child_nodes(
        self, child_nodes: List[Any], parent_class: Optional[TextClass] = None
    ) -> List[Dict[str, Any]]:
        """Helper method to process child nodes and collect results."""
        return [
            result
            for i, child in enumerate(child_nodes)
            for result in self.classify_latex_block(child, parent_class, i == 0)
        ]

    def parse(self, latex_file_path: Path) -> pd.DataFrame:
        """
        Parse a LaTeX file and extract structured paragraphs with labels.

        Args:
            latex_file_path: Path to the LaTeX file to parse

        Returns:
            DataFrame with parsed paragraphs and classification labels

        Raises:
            ValueError: If file path is invalid
            FileNotFoundError: If file doesn't exist
        """

        # Validate input
        if not isinstance(latex_file_path, Path):
            latex_file_path = Path(latex_file_path)

        if not latex_file_path.suffix == ".tex":
            raise ValueError(f"LaTeX file path must end with .tex: {latex_file_path}")

        if not latex_file_path.exists():
            raise FileNotFoundError(f"LaTeX file does not exist: {latex_file_path}")

        try:
            # Read the file
            with open(latex_file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            tex = TeX()
            tex.input(text)
            doc = tex.parse()
            rows = []
            for item in doc:
                if item is not None:
                    rows.extend(self.classify_latex_block(item))
            rows = [row for row in rows if row['text'] != '']
            for i, row in enumerate(rows):
                row['xml_idx'] = i

            # add line numbers
            for i, row in enumerate(rows):
                row['LineNumbers'] = i + 1

            # Convert list to DataFrame first
            df = pd.DataFrame(rows)
            df = df[['LineNumbers', 'text', 'PageNumber', 'SourceClass', 'xml_idx']]
            df['SourceClassName'] = df['SourceClass'].apply(lambda x: TextClass(x).name)
            
            return df

        except Exception as e:
            log.error("Error parsing LaTeX file: %s", e)
            traceback.print_exc()
            return pd.DataFrame({'text': []})


def main() -> None:
    """Main function for testing the parser."""
    # Example usage
    parser = LatexParser()

    # You would replace this with your actual file path
    data_dir = Path(__file__).parents[4] / "data" / "documents" / "Latex"
    latex_file_path = data_dir / "inspird.tex"

    if latex_file_path.exists():
        df = parser.parse(latex_file_path)

        # Save results
        output_dir = data_dir / "labelled_source"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = (output_dir / latex_file_path.name).with_suffix(".csv")

        df.to_csv(output_file, index=False)
        log.info("Parsed %s paragraphs from %s", len(df), latex_file_path)
        log.info("Results saved to %s", output_file)
    else:
        log.error("File not found: %s", latex_file_path)


if __name__ == "__main__":
    main()
