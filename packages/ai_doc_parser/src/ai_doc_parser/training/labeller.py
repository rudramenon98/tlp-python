"""
PDF labeling module for matching and classifying text lines between PDF and XML data.

This module provides functionality to match text lines from PDF documents with
corresponding XML annotations and assign classification labels based on content matching.
"""

import logging
import re
import time
import traceback

from enginius_parser.inference.feature_computation.feature_computer import first_word_compound

import unicodedata

# operator module no longer needed after refactoring get_page function
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import Levenshtein
import numpy as np
import pandas as pd
import unidecode

from enginius_parser.scripts.print_extracted_df_metrics import generate_metrics_report
from enginius_parser.training.common_tools import clean_text

try:
    from rapidfuzz import fuzz  # Much faster alternative to fuzzywuzzy
except ImportError:
    from fuzzywuzzy import fuzz  # Fallback to fuzzywuzzy if rapidfuzz not available

from enginius_parser.inference.text_class import TextClass
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class UsedTextTracker:
    """
    Tracks which portions of text have been used to prevent re-matching.
    """

    def __init__(self) -> None:
        # For each XML line, track which character ranges have been used
        self.used_ranges: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    def is_text_available(self, xml_idx: int, start_idx: int, end_idx: int) -> bool:
        """
        Check if a text range is available (not already used).

        Args:
            xml_idx: Index of the XML line
            start_idx: Start character index
            end_idx: End character index

        Returns:
            True if the text range is available, False otherwise
        """
        if xml_idx not in self.used_ranges:
            return True

        # Check if this range overlaps with any used ranges
        for used_start, used_end in self.used_ranges[xml_idx]:
            if not (end_idx <= used_start or start_idx >= used_end):
                # There's an overlap
                return False
        return True

    def mark_text_as_used(self, xml_idx: int, start_idx: int, end_idx: int) -> None:
        """
        Mark a text range as used.

        Args:
            xml_idx: Index of the XML line
            start_idx: Start character index
            end_idx: End character index
        """
        self.used_ranges[xml_idx].append((start_idx, end_idx))
        # Merge overlapping ranges
        self._merge_overlapping_ranges(xml_idx)

    def _merge_overlapping_ranges(self, xml_idx: int) -> None:
        """
        Merge overlapping ranges to keep the tracking efficient.
        """
        if xml_idx not in self.used_ranges:
            return

        ranges = sorted(self.used_ranges[xml_idx])
        merged: List[Tuple[int, int]] = []

        for start, end in ranges:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                # Merge with the last range
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))

        self.used_ranges[xml_idx] = merged

    def get_available_text(self, xml_idx: int, full_text: str) -> str:
        """
        Get the available (unused) portions of text for a given XML line.

        Args:
            xml_idx: Index of the XML line
            full_text: The full text of the XML line

        Returns:
            String containing only the unused portions of text
        """
        if xml_idx not in self.used_ranges:
            return full_text

        available_parts: List[str] = []
        last_end = 0

        for start, end in sorted(self.used_ranges[xml_idx]):
            if start > last_end:
                available_parts.append(full_text[last_end:start])
            last_end = end

        if last_end < len(full_text):
            available_parts.append(full_text[last_end:])

        return " ".join(available_parts)


def partial_match(regex: str, string: str) -> bool:
    """
    Check if a regex pattern partially matches a string using fuzzy matching.

    Args:
        regex: The regex pattern to match against
        string: The string to check

    Returns:
        True if partial ratio is greater than 80, False otherwise
    """
    ratio = fuzz.partial_ratio(regex, string)
    return ratio > 80


def matching(list1: List[str], list2: List[str]) -> float:
    """
    Calculate the probability of matching between two lists of strings.

    Args:
        list1: First list of strings to match
        list2: Second list of strings to match against

    Returns:
        Probability of successful matches (0.0 to 1.0)
    """
    if not list1 or not list2:
        return 0.0

    matching_count = 0

    # Pre-process list2 to avoid repeated string operations
    list2_lower = [str(x).lower() for x in list2]

    for item in list1:
        item_lower = item.lower()

        # Use early termination - stop as soon as we find a match
        for list2_item in list2_lower:
            if fuzz.partial_ratio(item_lower, list2_item) > 80:
                matching_count += 1
                break  # Early termination - no need to check other items

    return matching_count / len(list1)


def page_no(list1: List[float]) -> Optional[int]:
    """
    Find the index of the first value greater than 0.50 in a list.

    Args:
        list1: List of float values

    Returns:
        Index of first value > 0.50, or None if not found
    """
    for i, value in enumerate(list1):
        if value > 0.50:
            return i
    return None


def get_index(list1: List[str], list2: List[str], labels: List[str]) -> Dict[str, Optional[str]]:
    """
    Create a mapping from list1 items to their corresponding labels in list2.

    Args:
        list1: List of strings to find matches for
        list2: List of strings to search in
        labels: List of labels corresponding to list2

    Returns:
        Dictionary mapping list1 items to their corresponding labels
    """
    inputs = {}

    for item in list1:
        ids = None
        label = None

        for j, list2_item in enumerate(list2):
            if item in list2_item or fuzz.partial_ratio(list2_item, item) > 80:
                ids = j
                label = labels[ids]

        inputs[item] = ids
        inputs[item] = label

    return inputs


def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int = 100,
    overlap: int = 50,
    max_windows: int = 100,
) -> List[Tuple[int, int, int]]:
    """
    Create sliding windows from a dataframe based on word count.

    Args:
        df: DataFrame containing text data
        window_size: Target number of words in each window
        overlap: Target number of overlapping words between consecutive windows

    Returns:
        List of tuples (start_index, end_index) representing window boundaries
    """
    windows = []

    # Calculate word count for each row in the dataframe
    line_word_counts = []
    for _, row in df.iterrows():
        # Use the first column as text if it's a string, otherwise convert to string
        text = str(row['text'])
        word_count = len(text.split())
        line_word_counts.append(word_count)

    if len(line_word_counts) == 0:
        return windows

    start_idx = 0

    while start_idx < len(line_word_counts):
        current_word_count = 0
        current_idx = start_idx

        # Build window until we reach at least window_size words
        while current_idx < len(line_word_counts) and current_word_count < window_size:
            current_word_count += line_word_counts[current_idx]
            current_idx += 1

        # If we have a valid window (at least one row), add it
        if current_idx > start_idx:
            windows.append((start_idx, current_idx, current_word_count))

        # Find the next starting position by backing off from current position
        # to ensure overlap words are shared
        if current_idx >= len(line_word_counts):
            # We've reached the end, no more windows
            break

        # Calculate how many words we want to keep for overlap
        overlap_word_count = 0
        next_start_idx = current_idx

        # Work backwards from the end of current window to find overlap point
        for i in range(current_idx - 1, start_idx - 1, -1):
            if i >= 0 and i < len(line_word_counts):
                line_words = line_word_counts[i]
                if overlap_word_count + line_words <= overlap:
                    overlap_word_count += line_words
                    next_start_idx = i
                else:
                    break

        # Ensure we make progress (don't get stuck in infinite loop)
        if next_start_idx <= start_idx:
            next_start_idx = start_idx + 1

        start_idx = next_start_idx

    return windows


def calculate_window_similarity(
    pdf_window: List[str], xml_window: List[str], confidence_threshold: float = 0.7
) -> Tuple[float, List[float]]:
    """
    Calculate similarity between two windows using multiple metrics.
    Optimized for the case where PDF text is a subset of XML text with minor errors.

    Args:
        pdf_window: List of PDF lines in the window (smaller subset)
        xml_window: List of XML lines in the window (larger superset)

    Returns:
        Similarity score between 0 and 1
    """
    if not pdf_window or not xml_window:
        return 0.0, []
    pdf_text = " ".join(pdf_window)
    xml_text = " ".join(xml_window)
    return fuzz.partial_ratio(pdf_text, xml_text) / 100.0
    pdf_text_len = len(pdf_text.split())

    xml_line_lenths = [len(line.split()) for line in xml_window]
    xml_line_lenths_cumsum = np.cumsum(xml_line_lenths)
    max_score = 0
    for xml_start_idx in range(len(xml_window)):
        # calculate the xml_end_idx such that the length of text in the xml_window is closest to pdf_text_len
        xml_end_idx = np.argmin(np.abs(xml_line_lenths_cumsum - pdf_text_len - xml_line_lenths_cumsum[xml_start_idx]))
        xml_window_text = " ".join(xml_window[xml_start_idx : xml_end_idx + 1])
        if len(xml_window_text) < len(pdf_text) / 2:
            continue
        confidence = fuzz.partial_ratio(pdf_text, xml_window_text) / 100.0
        if confidence > confidence_threshold:
            return confidence
        if confidence > max_score:
            max_score = confidence
    return max_score


def clean_for_matching(text: str) -> str:
    """
    Clean text for matching while preserving word boundaries.
    Less aggressive than the original cleaning function.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text suitable for matching
    """
    if not text:
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFC", text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove some punctuation but keep word boundaries
    text = re.sub(r'[^\w\s]', ' ', text)

    # Clean up whitespace again
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def find_best_xml_window(
    pdf_window_bounds: Tuple[int, int],
    xml_window_bounds: List[Tuple[int, int]],
    pdf_df: pd.DataFrame,
    xml_df: pd.DataFrame,
    expected_xml_idx: int,
    max_windows_searched: int = 20,
    similarity_threshold: float = 0.7,
) -> Tuple[int, float]:
    """
    Find the best matching XML window for a given PDF window using optimized search.

    Args:
        pdf_window_bounds: Tuple of (start_index, end_index) representing the PDF window
        xml_window_bounds: List of tuples of (start_index, end_index) representing the XML windows
        pdf_df: DataFrame containing the PDF data
        xml_df: DataFrame containing the XML data
        expected_xml_idx: Expected starting position for the XML window (if known)
        max_windows_searched: Maximum number of windows to search before stopping
        similarity_threshold: Minimum similarity score to consider a match

    Returns:
        Tuple of (best_xml_start_index, best_similarity_score)
    """
    best_score = 0.0
    best_xml_idx = -1

    pdf_start, pdf_end = pdf_window_bounds
    pdf_window = pdf_df.iloc[pdf_start:pdf_end]['cleaned_text'].tolist()

    search_start = max(0, expected_xml_idx - max_windows_searched // 2)
    search_end = min(len(xml_window_bounds), expected_xml_idx + max_windows_searched // 2)
    search_indices = list(range(search_start, search_end))
    search_indices = sorted(search_indices, key=lambda x: abs(x - expected_xml_idx))

    # Search through the prioritized indices
    for xml_i in search_indices:
        if xml_i >= len(xml_window_bounds):
            continue

        xml_start, xml_end, _ = xml_window_bounds[xml_i]
        # Extract the actual XML lines for this window
        xml_window = xml_df.iloc[xml_start:xml_end]['cleaned_text'].tolist()

        # Calculate direct window similarity
        direct_similarity = calculate_window_similarity(pdf_window, xml_window)

        # Early termination if we find a very good match
        # if direct_similarity > similarity_threshold:
        #     best_xml_idx = xml_i
        #     return best_xml_idx, direct_similarity
        if direct_similarity > best_score:
            best_score = direct_similarity
            best_xml_idx = xml_i
        if direct_similarity > similarity_threshold:
            return xml_i, direct_similarity

    return best_xml_idx, best_score


def sentence_similarity(s1: str, s2: str) -> float:
    dist = Levenshtein.distance(s1.strip(), s2.strip())
    partial_ratio = 1 - dist / max(len(s1), len(s2))
    return partial_ratio


def match_line_to_block(
    pdf_line: str,
    source_block: str,
    text_tracker: Optional[UsedTextTracker] = None,
    xml_idx: Optional[int] = None,
    score_threshold: float = 1.0,
) -> Tuple[float, Tuple[int, int]]:
    """
    Find where a pdf_line matches best within a source_block.

    Args:
        pdf_line: The PDF line to match
        source_block: The source block to search in
        text_tracker: Optional tracker for used text segments
        xml_idx: Index of the XML line (required if text_tracker is provided)

    Returns:
        Tuple[float, Tuple[int, int]]: Tuple containing the confidence score and the start and end indices of the matched block
    """
    index_mapping: Optional[List[int]] = None

    # If text tracker is provided, only search in available (unused) portions
    if text_tracker is not None and xml_idx is not None:
        available_text = text_tracker.get_available_text(xml_idx, source_block)
        if not available_text.strip():
            # No available text to match against
            return 0.0, (0, 0)

        # Create a mapping from available text indices to original text indices
        index_mapping = []
        available_parts = []
        last_end = 0

        for start, end in sorted(text_tracker.used_ranges[xml_idx]):
            if start > last_end:
                available_parts.append(source_block[last_end:start])
                index_mapping.extend(range(last_end, start))
            last_end = end

        if last_end < len(source_block):
            available_parts.append(source_block[last_end:])
            index_mapping.extend(range(last_end, len(source_block)))

        source_block = " ".join(available_parts)

    if len(pdf_line) > len(source_block):
        score = sentence_similarity(pdf_line, source_block)
        if index_mapping and len(index_mapping) > 0:
            return score, (index_mapping[0], index_mapping[-1] + 1)
        return score, (0, len(source_block))

    if pdf_line in source_block:
        start_index = source_block.index(pdf_line)
        if index_mapping and start_index < len(index_mapping):
            original_start = index_mapping[start_index]
            original_end = index_mapping[min(start_index + len(pdf_line) - 1, len(index_mapping) - 1)] + 1
            return 1.0, (original_start, original_end)
        return 1.0, (start_index, start_index + len(pdf_line))

    source_words = source_block.split()
    num_leading_spaces = len(source_block) - len(source_block.lstrip())
    word_idxs = [num_leading_spaces]
    for word in source_words[:-1]:
        word_idx = word_idxs[-1] + len(word) + 1 + num_leading_spaces
        word_idxs.append(word_idx)

    scores = []

    for word_idx in word_idxs:
        source_line = source_block[word_idx : word_idx + len(pdf_line)]
        score = sentence_similarity(pdf_line, source_line)
        scores.append(score)
        if score > score_threshold:
            break

    best_idx = np.argmax(scores)
    best_score = scores[best_idx]

    # Map indices back to original text if needed
    if index_mapping and best_idx < len(word_idxs):
        start_in_available = word_idxs[best_idx]
        end_in_available = word_idxs[best_idx] + len(pdf_line)

        if start_in_available < len(index_mapping) and end_in_available <= len(index_mapping):
            original_start = index_mapping[start_in_available]
            original_end = index_mapping[end_in_available - 1] + 1
            return best_score, (original_start, original_end)

    start_idx = word_idxs[best_idx] + num_leading_spaces
    return best_score, (start_idx, start_idx + len(pdf_line))


def is_toc(line: str) -> bool:
    line = line.lower()
    line = line.replace(' ', '')
    # if there are more than 5 consecutive dots, then it is a table of contents
    if re.search(r'\.{5,}', line):
        return True
    # if there are more than 5 consecutive '-', then it is a table of contents
    if re.search(r'-{5,}', line):
        return True
    # if there are more than 5 consecutive '_'
    if re.search(r'_{5,}', line):
        return True
    return False


def match_lines_within_windows(
    pdf_df: pd.DataFrame,
    xml_df: pd.DataFrame,
    pdf_windows: List[Tuple[int, int]],
    xml_windows: List[Tuple[int, int]],
    pdf_window_idx: int,
    xml_window_idx: int,
    text_tracker: Optional[UsedTextTracker] = None,
    window_search_size: int = 3,
    best_xml_idx: int = -1,
) -> Tuple[pd.DataFrame, int]:
    """
    Match individual lines within matched windows.

    Args:
        pdf_df: DataFrame containing the PDF data
        xml_df: DataFrame containing the XML data
        pdf_windows: List of tuples of (start_index, end_index) representing the PDF windows
        xml_windows: List of tuples of (start_index, end_index) representing the XML windows
        pdf_window_idx: Index of the PDF window
        xml_window_idx: Index of the XML window
        text_tracker: Optional tracker for used text segments
        window_search_size: Size of the search window for XML lines
        best_xml_idx: Index of the best XML line found in the previous window

    Returns:
        DataFrame with matched lines containing columns: pdf_idx, xml_idx, confidence, assigned_class, matches_beginning
    """
    matches = []

    # Get range of xml_lines to be matched to
    start_window_idx = max(xml_window_idx - window_search_size, 0)
    end_window_idx = min(xml_window_idx + window_search_size, len(xml_windows) - 1)
    xml_start_idx = xml_windows[start_window_idx][0]
    xml_end_idx = xml_windows[end_window_idx][1]

    # Ensure the search range doesn't exceed the DataFrame bounds
    xml_start_idx = max(0, min(xml_start_idx, len(xml_df)))
    xml_end_idx = max(0, min(xml_end_idx, len(xml_df)))

    xml_search_range = range(xml_start_idx, xml_end_idx)

    last_best_xml_idx = best_xml_idx

    # Match each PDF line to the best XML line
    pdf_window_start_idx, pdf_window_end_idx, _ = pdf_windows[pdf_window_idx]
    for pdf_df_idx in range(pdf_window_start_idx, pdf_window_end_idx):
        pdf_row = pdf_df.iloc[pdf_df_idx]
        if not pd.isna(pdf_row['ExtractedClass']):
            continue

        pdf_line = pdf_row['text']
        pdf_cleaned = pdf_row['cleaned_text']
        pdf_word_count = len(pdf_line.split())

        if is_toc(pdf_line):
            # Store the match with the assigned class
            labelled_row = pdf_row.copy()
            labelled_row['LabelledClass'] = TextClass.TOC.value
            labelled_row['XML_text'] = ""
            labelled_row['Match_Confidence'] = 0
            labelled_row['pdf_window_idx'] = pdf_window_idx
            labelled_row['xml_window_idx'] = 0
            labelled_row['LabelledClassName'] = TextClass.TOC.name
            labelled_row['xml_idx'] = -1
            labelled_row['pdf_idx'] = pdf_row['pdf_idx']
            labelled_row['xml_start_idx'] = 0
            labelled_row['xml_end_idx'] = 0
            labelled_row['source_matched_text'] = ""
            labelled_row['pdf_cleaned'] = pdf_cleaned
            matches.append(labelled_row)
            continue
        if pdf_line == 'nan' or not pdf_line.strip() or len(pdf_line) <= 1:
            continue
        best_xml_idx = -1
        best_confidence = 0.0
        matches_beginning = False

        pdf_cleaned = pdf_df.iloc[pdf_df_idx]['cleaned_text']
        best_start_idx = 0
        best_end_idx = 0
        best_matching_text = ""

        best_xml_row = None
        # Custom sorting: second element > first, third element < first
        xml_search_range = list(range(xml_start_idx, xml_end_idx))
        xml_search_range.sort(key=lambda x: (abs(last_best_xml_idx - x), x > last_best_xml_idx, x))

        for xml_idx in xml_search_range[:100]:
            # Check if xml_idx is within bounds of xml_df
            if xml_idx >= len(xml_df):
                continue
            xml_row = xml_df.iloc[xml_idx]
            if xml_row['SourceClass'] in [TextClass.TABLE, TextClass.HEADER, TextClass.FOOTER]:
                continue
            xml_cleaned = xml_row['cleaned_text']
            # xml_idx = xml_row['index']
            available_text = text_tracker.get_available_text(xml_idx, xml_cleaned)
            if available_text.strip() == "":
                continue

            # Fast exact match check first
            if pdf_cleaned != "" and pdf_cleaned in available_text:
                confidence = 1.0
                start_idx = xml_cleaned.index(pdf_cleaned)
                end_idx = start_idx + len(pdf_cleaned)
            else:
                # Calculate similarity only for promising candidates
                confidence, (start_idx, end_idx) = match_line_to_block(pdf_cleaned, xml_cleaned, text_tracker, xml_idx)

                # if confidence is low and first word is compound, try to match without first word
                if first_word_compound({"text": pdf_cleaned}) and len(pdf_cleaned.split(" ")) > 1:
                    pdf_cleaned_without_first_word = " ".join(pdf_cleaned.split(" ")[1:])
                    new_confidence, (new_start_idx, new_end_idx) = match_line_to_block(
                        pdf_cleaned_without_first_word, xml_cleaned, text_tracker, xml_idx
                    )
                    if new_confidence > confidence:
                        confidence = new_confidence
                        start_idx = new_start_idx
                        end_idx = new_end_idx

                    # pdf_cleaned = pdf_cleaned_without_first_word

                # # Penalty for significant length differences
                # if len(pdf_cleaned) > len(xml_cleaned):
                #     confidence = min(confidence, 0.5)

            if confidence > best_confidence:
                best_confidence = confidence
                best_xml_row = xml_row
                best_xml_idx = xml_idx
                best_start_idx = start_idx
                best_end_idx = end_idx
                best_matching_text = xml_cleaned[start_idx:end_idx]
                # Check if PDF line matches the beginning of XML line
                matches_beginning = start_idx < len(xml_cleaned.split(" ")[0])

                # Early exit for perfect matches
                if confidence > 0.8:
                    break

        if best_confidence < 0.75:
            continue

        if best_xml_idx == -1:
            log.warning("Could not match pdf line %s", pdf_line)
            continue

        # Mark the matched text as used in the text tracker
        if text_tracker is not None and best_xml_idx != -1 and best_xml_idx < len(xml_df):
            # Mark the matched range as used
            if abs(best_xml_idx - last_best_xml_idx) < 10:
                text_tracker.mark_text_as_used(best_xml_idx, best_start_idx, best_end_idx)
        last_best_xml_idx = best_xml_idx

        if not pd.isna(pdf_row['ExtractedClass']):
            assigned_class = pdf_row['ExtractedClass']
        else:
            # Get the XML class for this line
            xml_class = xml_df.iloc[best_xml_idx]['SourceClass'] if best_xml_idx < len(xml_df) else 0

            # Apply class assignment logic based on beginning match and XML class
            assigned_class = xml_class
            if xml_class == TextClass.PARAGRAPH:
                if not matches_beginning:
                    assigned_class = TextClass.PARAGRAPH_CONT
            if xml_class == TextClass.HEADING:
                if not matches_beginning:
                    assigned_class = TextClass.HEADING_CONT
            if xml_class == TextClass.ENUM_LIST:
                if not matches_beginning:
                    assigned_class = TextClass.ENUM_LIST_CONT
            if xml_class == TextClass.BULLET_LIST:
                if not matches_beginning:
                    assigned_class = TextClass.BULLET_LIST_CONT

        # Store the match with the assigned class
        labelled_row = pdf_row.copy()
        labelled_row['LabelledClass'] = assigned_class
        # Add bounds checking for best_xml_idx
        if best_xml_idx < len(xml_df):
            # labelled_row['XML_line_Number'] = xml_df.iloc[best_xml_idx]['LineNumbers']
            labelled_row['XML_text'] = xml_df.iloc[best_xml_idx]['text']
        else:
            # labelled_row['XML_line_Number'] = 0
            labelled_row['XML_text'] = ""
        labelled_row['Match_Confidence'] = best_confidence
        labelled_row['pdf_window_idx'] = pdf_window_idx
        labelled_row['xml_window_idx'] = xml_window_idx
        labelled_row['LabelledClassName'] = TextClass(assigned_class).name
        labelled_row['xml_idx'] = best_xml_row['xml_idx']
        labelled_row['pdf_idx'] = pdf_row['pdf_idx']
        labelled_row['xml_start_idx'] = best_start_idx
        labelled_row['xml_end_idx'] = best_end_idx
        labelled_row['source_matched_text'] = best_matching_text
        labelled_row['pdf_cleaned'] = pdf_cleaned
        matches.append(labelled_row)

    window_matches_df = pd.DataFrame(matches)
    return window_matches_df, last_best_xml_idx


def sliding_window_labeling(
    pdf_df: pd.DataFrame, xml_df: pd.DataFrame, window_size: int = 2000, overlap: int = 1500
) -> pd.DataFrame:
    """
    Perform labeling using sliding window approach without page numbers.

    Args:
        pdf_df: DataFrame containing PDF data
        xml_df: DataFrame containing XML data
        window_size: Target number of words in each sliding window
        overlap: Target number of overlapping words between windows

    Returns:
        DataFrame with added ClassLabel and XML_line_Number columns
    """
    # drop rows with no text or nan
    pdf_df = pdf_df[pdf_df['text'].notna() & (pdf_df['text'] != '')]
    xml_df = xml_df[xml_df['text'].notna() & (xml_df['text'] != '')]
    xml_df = xml_df.sort_values(['xml_idx'], ascending=True)

    # Prepare data
    pdf_lines = list(pdf_df['text'])
    pdf_lines = [str(line) for line in pdf_lines]

    xml_lines = list(xml_df['text'])
    xml_lines = [str(line) for line in xml_lines]

    # add cleaned_text column to dataframes
    log.debug("Cleaning XML lines")
    pdf_df['cleaned_text'] = [clean_text(line) for line in pdf_lines]
    xml_df['cleaned_text'] = [clean_text(line) for line in xml_lines]

    # Handle edge cases
    if not pdf_lines:
        log.warning("No PDF lines found")
        result_df = pdf_df.copy()
        result_df['ClassLabel'] = 0
        result_df['XML_line_Number'] = np.nan
        result_df['Match_Confidence'] = 0.0
        return result_df

    if not xml_lines:
        log.warning("No XML lines found")
        result_df = pdf_df.copy()
        result_df['ClassLabel'] = 0
        result_df['XML_line_Number'] = np.nan
        result_df['Match_Confidence'] = 0.0
        return result_df

    # Calculate total words in each dataset
    pdf_total_words = sum(len(line.split()) for line in pdf_lines)
    xml_total_words = sum(len(line.split()) for line in xml_lines)

    # Adjust window size if needed
    if window_size > pdf_total_words:
        window_size = max(50, pdf_total_words // 4)
        log.debug("Adjusted window size to %s words due to small PDF", window_size)

    if window_size > xml_total_words:
        window_size = max(50, xml_total_words // 4)
        log.debug("Adjusted window size to %s words due to small XML", window_size)

    # remove tables, headers, footers
    xml_df = xml_df[~xml_df['SourceClass'].isin([TextClass.TABLE, TextClass.HEADER, TextClass.FOOTER])]
    pdf_df = pdf_df[~pdf_df['ExtractedClass'].isin([TextClass.TABLE, TextClass.HEADER, TextClass.FOOTER])]

    pdf_windows = create_sliding_windows(pdf_df, 200, 1)
    xml_windows = create_sliding_windows(xml_df, 800, 300)

    log.debug("Created %s PDF windows and %s XML windows", len(pdf_windows), len(xml_windows))

    # Initialize text tracker to prevent re-matching of used text segments
    text_tracker = UsedTextTracker()

    expected_xml_idx = 0
    start_time = time.time()
    # Process each PDF window
    best_xml_idx = -1

    pdf_total_words = sum([pdf_word_count for _, _, pdf_word_count in pdf_windows])

    # List to collect all match DataFrames
    all_matches_dfs = []
    for pdf_window_idx, (pdf_start, pdf_end, pdf_word_count) in enumerate(pdf_windows):
        if not (min_pdf_window <= pdf_window_idx <= max_pdf_window):
            continue
        time_elapsed = time.time() - start_time
        time_remaining_sec = (len(pdf_windows) - pdf_window_idx - 1) * time_elapsed / (pdf_window_idx + 1)
        time_remaining_formatted = time.strftime("%H:%M:%S", time.gmtime(time_remaining_sec))

        # Find best matching XML window with optimized search
        xml_window_idx, similarity_score = find_best_xml_window(
            (pdf_start, pdf_end),
            xml_windows,
            pdf_df,
            xml_df,
            expected_xml_idx,
            max_windows_searched=5,
        )

        if similarity_score < 0.7:
            pdf_word_count = sum([pdf_word_count for _, _, pdf_word_count in pdf_windows[:pdf_window_idx]])
            expected_xml_idx = int(pdf_word_count / pdf_total_words * len(xml_windows))
            wider_xml_window_idx, wider_similarity_score = find_best_xml_window(
                (pdf_start, pdf_end),
                xml_windows,
                pdf_df,
                xml_df,
                expected_xml_idx,
                max_windows_searched=50,  # Limit search to 15 windows for speed
            )
            if wider_similarity_score > similarity_score:
                xml_window_idx = wider_xml_window_idx
                similarity_score = wider_similarity_score

        if similarity_score < 0.5:
            log.warning("BAD MATCH")
            xml_window_idx = expected_xml_idx
        log.debug(
            f"Matched PDF window {pdf_window_idx}/{len(pdf_windows)} to XML window {xml_window_idx} (score: {similarity_score:.3f}) Expected: {expected_xml_idx} Time remaining: {time_remaining_formatted}"
        )

        expected_xml_idx = xml_window_idx
        # Match individual lines within the windows
        window_matches_df, best_xml_idx = match_lines_within_windows(
            pdf_df,
            xml_df,
            pdf_windows,
            xml_windows,
            pdf_window_idx,
            xml_window_idx,
            text_tracker=text_tracker,
            best_xml_idx=best_xml_idx,
        )

        # Find which window contains the best_xml_idx
        expected_xml_idx = None
        for i, window in enumerate(xml_windows):
            if window[0] <= best_xml_idx < window[1]:
                expected_xml_idx = i
                break

        # Add window information to the matches DataFrame
        if not window_matches_df.empty:
            all_matches_dfs.append(window_matches_df)

    # Concatenate all match DataFrames and apply to result_df
    if all_matches_dfs:
        result_df = pd.concat(all_matches_dfs, ignore_index=True)
        # Remove duplicates based on pdf_idx (keep the first occurrence)
        result_df = result_df.drop_duplicates(subset=['pdf_idx'], keep='first')

    matched_pdf_indices = set(result_df['pdf_idx'])

    # Add summary statistics
    total_lines = len(pdf_lines)
    matched_lines = len(matched_pdf_indices)
    match_percentage = (matched_lines / total_lines) * 100 if total_lines > 0 else 0

    log.info("Matched %s lines out of %s total lines (%.1f%%)", matched_lines, total_lines, match_percentage)

    # Calculate average confidence from the result DataFrame
    matched_confidences = result_df[result_df['Match_Confidence'] > 0]['Match_Confidence']
    if len(matched_confidences) > 0:
        log.debug(f"Average confidence: {matched_confidences.mean():.3f}")
    else:
        log.debug("No matches found")

    # set xml_line_number column to be first column
    columns = result_df.columns.tolist()
    # move xml_line_number to first column
    result_df = result_df[columns]
    return result_df


def main() -> None:
    from enginius_parser import EASA_PDF as pdf_path
    from enginius_parser import EASA_SOURCE as source_path

    pdf_path = Path(r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\Latex\v28.pdf")
    source_path = Path(r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\Latex\inspird.tex")
    pdf_df_path = pdf_path.parent / "computed_features" / f"{pdf_path.stem}.csv"
    xml_df_path = source_path.parent / "labelled_source" / f"{source_path.stem}.csv"
    xml_df_path = (
        r"C:\Users\r123m\Documents\enginius\source\ai-pdf-parser\data\documents\Latex\labelled_source\inspird.csv"
    )
    document_dir = pdf_path.parent

    # pdf_df_path = DATA_DIR / "documents" / "CFR" / "computed_features" / "CFR-2024-title21-vol8-chapI-subchapH.csv"
    # xml_df_path = DATA_DIR / "documents" / "CFR" / "labelled_source" / "CFR-2024-title21-vol8-chapI-subchapH.csv"

    pdf_df = pd.read_csv(pdf_df_path)
    xml_df = pd.read_csv(xml_df_path)

    # Use the new sliding window approach instead of page-based matching
    log.info("Using sliding window labeling approach...")
    output_file = document_dir / "labelled_pdf" / f"{pdf_df_path.stem}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df = sliding_window_labeling(pdf_df, xml_df)
    final_df.to_csv(output_file, index=False)

    log.info("Labeling completed successfully!")
    log.info("Saved to %s", output_file)

    extracted_labels_path = document_dir / "labelled_source" / f"{source_path.stem}.csv"
    labelled_pdf_path = document_dir / "labelled_pdf" / f"{pdf_path.stem}.csv"
    classified_pdf_path = document_dir / "ai_parsed_pdf" / f"{pdf_path.stem}.csv"
    generate_metrics_report(extracted_labels_path, labelled_pdf_path, classified_pdf_path)


min_pdf_window = 0
max_pdf_window = np.inf
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # source_block = "(1) Except as provided in (c) (1) below, for surveying purposes the centre (mid-width) of the taxiway centre line marking, apron taxilane marking or the aircraft stand guide Intersection of centre of lead-in line marking and centre of taxilane markingline marking must be taken as the reference data."
    # pdf_line = "H Intersection of centre of lead-in line marking and centre of taxilane marking"

    # score, (start_idx, end_idx) = match_line_to_block(pdf_line, source_block)
    # print(f"Score: {score}, Start index: {start_idx}, End index: {end_idx}")
    # print(pdf_line)
    # print(source_block[start_idx:end_idx])

    min_pdf_window = 0
    # max_pdf_window = 20
    main()
