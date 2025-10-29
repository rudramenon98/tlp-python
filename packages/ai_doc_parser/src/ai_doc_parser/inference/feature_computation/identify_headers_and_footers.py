"""@author: dshah"""

import collections
import itertools
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def get_mode(array: List[float]) -> float:
    """
    Calculate the mode of an array with tolerance for floating point values.

    Args:
        array: List of numeric values

    Returns:
        The most frequently occurring value within tolerance
    """
    epsilon = 2

    def almost_equal(a: float, b: float) -> bool:
        return -epsilon <= a - b <= epsilon

    # Group similar values together with their counts
    counts = []
    for a in array:
        for i, (b, c) in enumerate(counts):
            if almost_equal(a, b):
                counts[i] = (b, c + 1)
                break
        else:
            counts.append((a, 1))

    # Find the most frequent value
    if len(counts) > 0:
        mode = sorted(counts, key=lambda k: k[1])[-1][0]
    else:
        mode = 0
    return mode


def get_mean(array: List[float]) -> float:
    """
    Calculate the mean of an array.

    Args:
        array: List of numeric values

    Returns:
        The mean value rounded to 2 decimal places
    """
    num_elements = len(array)

    if num_elements == 0:
        return 0

    sum_val = 0
    for a in array:
        sum_val += a

    return round(sum_val / num_elements, 2)


def ordered_cluster(data: List[float], max_diff: float) -> List[Tuple[float, ...]]:
    """
    Group consecutive data points that are within max_diff of their mean.

    Args:
        data: List of numeric values
        max_diff: Maximum difference allowed within a cluster

    Yields:
        Tuples of clustered values
    """
    current_group = ()
    for item in data:
        test_group = current_group + (item,)
        test_group_mean = mean(test_group)
        if all((abs(test_group_mean - test_item) < max_diff for test_item in test_group)):
            current_group = test_group
        else:
            yield current_group
            current_group = (item,)
    if current_group:
        yield current_group


def get_ranges(x: List[float], tolerance: float) -> List[List[int]]:
    """
    Find ranges of consecutive values that are within tolerance of each other.

    Args:
        x: List of numeric values
        tolerance: Maximum difference allowed between consecutive values

    Returns:
        List of [start_index, end_index] ranges
    """
    out = []
    count = 0
    for i in range(1, len(x)):
        prev = x[i - 1]
        if abs(x[i] - prev) <= tolerance:
            count += 1
        elif count and abs(x[i] - prev) > tolerance:
            out.append([i - count - 1, i - 1])
            count = 0
        else:
            out.append([i - count - 1, i - 1])
    if count:
        out.append([i - count, i])
    else:
        out.append([i - count, i])

    return out


def check_footnote(row: pd.Series, bottom_margin: float, bottom_margin2: float) -> bool:
    """
    Check if a line is a footnote based on position, length, and page orientation.

    Args:
        row: pandas Series containing line data with columns:
             - line_y1: y-coordinate of the line
             - line_len: length of the line text
             - page_width: width of the page
             - page_height: height of the page
        bottom_margin: Bottom margin for portrait orientation
        bottom_margin2: Bottom margin for landscape orientation

    Returns:
        True if line appears to be a footnote
    """
    # Portrait orientation check: line is below bottom margin and has short text
    y1 = row['line_y0'] + float(row['major_font_size'])
    portrait_footnote = y1 >= bottom_margin and row['line_len'] < row['page_width']

    # Landscape orientation check: page is wider than tall and line is below bottom margin
    landscape_footnote = row['page_width'] > row['page_height'] and y1 >= bottom_margin2

    return portrait_footnote or landscape_footnote


def calculate_top_margin(modal_line_space: List[float], modal_y_coordinate_top: float) -> float:
    """
    Calculate the top margin based on line spacing patterns.

    Args:
        modal_line_space: List of modal line spacing values
        modal_y_coordinate_top: Modal y-coordinate for top of pages

    Returns:
        Calculated top margin value
    """
    top_margin = 0
    max_top_line_space = max(modal_line_space)
    max_top_line_space_index = modal_line_space.index(max_top_line_space)

    # If maximum line spacing occurs after first few lines, use top coordinate
    if max_top_line_space_index > 2:
        top_margin = modal_y_coordinate_top
    else:
        top_ranges = get_ranges(modal_line_space, 2)
        if len(top_ranges) == 1:  # all values are similar -> no header
            top_margin = modal_y_coordinate_top
        elif len(top_ranges) > 1:
            # Calculate header height based on spacing ranges
            header_start = top_ranges[0][0]
            header_end = top_ranges[1][0]

            top_margin += modal_y_coordinate_top
            for i in range(header_start, header_end):
                top_margin = top_margin + modal_line_space[i]
            top_margin -= modal_line_space[header_end] // 2

    return top_margin


def identify_headers_and_footnotes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify headers and footnotes in a PDF document by analyzing text positioning.

    Args:
        df: DataFrame containing PDF text data with columns like 'text', 'line_y0', 'line_y1', etc.

    Returns:
        DataFrame with added 'is_header' and 'is_footnote' boolean columns
    """
    # Initialize arrays to store line spacing data for different page positions
    all_page_line_space = [[] for _ in range(7)]
    all_pages_all_line_space = [[] for _ in range(6)]
    all_y_coordinate_top = []
    all_y_coordinate_bottom = []
    all_y_coordinate_bottom2 = []

    # Initialize variables for calculating line spacing statistics
    sum_line_space = 0
    all_line_space = []
    num_lines = 0

    # Add line length column for analysis
    df['line_len'] = df['text'].str.len()

    # Group data by page number for page-wise processing
    groups = df.groupby('PageNumber')

    for _, each_file_json in groups:
        # Sort text blocks from top to bottom on the page
        each_file_json.sort_values(by='line_y0', inplace=True)

        # Initialize page-specific spacing array
        page_line_space = []

        # Get page dimensions
        page_width = each_file_json.iloc[0]['page_width']
        page_height = each_file_json.iloc[0]['page_height']

        # Determine page orientation (0=portrait, 1=landscape)
        page_orientation = 0 if page_height > page_width else 1

        count = 0

        # Process each line on the page
        for i in range(each_file_json.shape[0]):
            if i < each_file_json.shape[0] - 1:
                currRow = each_file_json.iloc[i]
                nextRow = each_file_json.iloc[i + 1]

                # Calculate line spacing between current and next line
                line_space = abs(float(currRow["line_y0"]) - float(nextRow["line_y0"]))

                # Skip very small line spaces (likely same line)
                if line_space < 7:
                    continue

                page_line_space.append(round(line_space))
                sum_line_space += line_space
                all_line_space.append(line_space)
                num_lines += 1

                # Store line spacing for first 6 lines of each page
                if count < 7:
                    all_page_line_space[count].append(round(line_space))

                # Store y-coordinate of first line (top of page)
                if count == 0:
                    all_y_coordinate_top.append(round(float(currRow["line_y0"])))

            count += 1

        max_y = max(each_file_json['line_y0'])
        # Store bottom y-coordinate based on page orientation
        if page_orientation == 0:  # portrait
            all_y_coordinate_bottom.append(round(max_y, 2))
        else:
            all_y_coordinate_bottom2.append(round(max_y, 2))

        # Store last 5 line spaces from each page
        for i in [1, 2, 3, 4, 5]:
            try:
                all_pages_all_line_space[i - 1].append(round(page_line_space[-i]))
            except IndexError:
                pass

    # Calculate modal line spacing for different page positions
    modal_line_space = []
    for ls in all_page_line_space:
        mode = get_mode(ls)
        modal_line_space.append(mode)

    # Calculate modal line spacing for bottom analysis
    modal_all_line_space = []
    for ls in all_pages_all_line_space:
        mode2 = get_mode(ls)
        modal_all_line_space.append(mode2)

    # Calculate modal y-coordinates for top and bottom of pages
    modal_y_coordinate_top = get_mode(all_y_coordinate_top)
    modal_y_coordinate_bottom = get_mode(all_y_coordinate_bottom)
    modal_y_coordinate_bottom2 = get_mode(all_y_coordinate_bottom2)

    # Use portrait bottom coordinate if landscape coordinate is not available
    if modal_y_coordinate_bottom2 == 0:
        modal_y_coordinate_bottom2 = modal_y_coordinate_bottom

    # Calculate top margin based on line spacing patterns
    top_margin = calculate_top_margin(modal_line_space, modal_y_coordinate_top)

    # Mark lines above top margin as headers
    df['is_header'] = np.where((df['line_y1'] + df['line_y0']) / 2 >= top_margin, False, True)
    df['top_margin'] = top_margin

    # Calculate bottom margins for both orientations
    bottom_margin = modal_y_coordinate_bottom
    bottom_margin2 = modal_y_coordinate_bottom2
    df['bottom_margin'] = bottom_margin

    # Initialize footnote column
    df['is_footnote'] = False

    # Apply footnote detection
    df['is_footnote'] = df.apply(lambda row: check_footnote(row, bottom_margin, bottom_margin2), axis=1)

    # Remove temporary line length column
    df.drop(['line_len'], axis=1, inplace=True)

    # Convert coordinate columns to integers for consistency
    df['block_x0'] = df['block_x0'].apply(lambda x: int(x))
    df['block_y0'] = df['block_y0'].apply(lambda x: int(x))
    df['block_x1'] = df['block_x1'].apply(lambda x: int(x))
    df['block_y1'] = df['block_y1'].apply(lambda x: int(x))
    df['line_x0'] = df['line_x0'].apply(lambda x: int(x))
    df['line_y0'] = df['line_y0'].apply(lambda x: int(x))
    df['line_x1'] = df['line_x1'].apply(lambda x: int(x))
    df['line_y1'] = df['line_y1'].apply(lambda x: int(x))

    return df
