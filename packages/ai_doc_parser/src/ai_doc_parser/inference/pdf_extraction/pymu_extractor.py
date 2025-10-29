import itertools
import logging
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
import numpy as np
import pandas as pd
import unidecode
from pymupdf.table import TableFinder

from ai_doc_parser.inference.feature_computation.feature_methods import mergeIntervals

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def calculate_iou(rect1: Tuple[float, float, float, float], rect2: Tuple[float, float, float, float]) -> float:
    """
    Calculates the percent overlap between two rectangles using:
    (2 * intersection_area) / (area_rect1 + area_rect2)

    Args:
        rect1: (x1, y1, x2, y2) for first rectangle
        rect2: (x1, y1, x2, y2) for second rectangle
               Coordinates should follow: (left, top, right, bottom)
    Returns:
        Percentage overlap as a float in [0, 1].
    """

    # Coordinates of intersection
    inter_left = max(rect1[0], rect2[0])
    inter_top = max(rect1[1], rect2[1])
    inter_right = min(rect1[2], rect2[2])
    inter_bottom = min(rect1[3], rect2[3])

    # Intersection area
    inter_width = max(0.0, inter_right - inter_left)
    inter_height = max(0.0, inter_bottom - inter_top)
    inter_area = inter_width * inter_height

    # Areas of both rectangles
    area1 = max(0.0, rect1[2] - rect1[0]) * max(0.0, rect1[3] - rect1[1])
    area2 = max(0.0, rect2[2] - rect2[0]) * max(0.0, rect2[3] - rect2[1])

    if area1 == 0 or area2 == 0:
        return 0.0

    # Modified IoU formula
    return (2.0 * inter_area) / (area1 + area2)


def header_footnote(Class, header, footnote):
    # print(Class,header,footnote)
    if not Class or str(Class) == "nan":
        if header == True:
            # print("here")
            return 6
        if footnote == True:
            # print("here")
            return 7
    else:
        return Class


def merge_dict(dict1: Dict, dict2: Dict) -> Dict:
    """Merge dictionaries and keep values of common keys in list"""

    dict3 = {**dict1, **dict2}

    for key, value in dict3.items():

        if key in dict1 and key in dict2:
            dict3[key] = value + dict1[key]

    return dict3


def is_table_content(
    table_coordinate: List[Tuple[float, float, float, float]],
    bx0: float,
    bx1: float,
    by0: float,
    by1: float,
) -> bool:
    # check if a line coordinate is inside a table coordinate
    bx0, bx1, by0, by1 = map(float, (bx0, bx1, by0, by1))
    for table_coor in table_coordinate:
        tx0, ty0, tx1, ty1 = map(float, table_coor)
        if (tx0 < bx0 < tx1 or tx0 < bx1 < tx1) and (ty0 < by0 < ty1 or ty0 < by1 < ty1):
            return True
    return False


def line_info(block: ET.Element) -> Tuple[str, str, str, str, str, str]:
    # return info of a line including font_family and font_size,
    # and coordinate of left most char in the line
    # the process is done only for the first detected font in the line
    for font in block:
        for char in font:
            font_family = font.attrib["name"]
            font_size = font.attrib["size"]
            most_left_x = block.attrib["bbox"].split(" ")[0]
            most_left_y = block.attrib["bbox"].split(" ")[1]
            x0 = char.attrib["x"]
            y0 = char.attrib["y"]
            return font_family, font_size, most_left_x, most_left_y, x0, y0
            break
        break


def return_line_of_text(line) -> str:
    # return a line of text in PDF
    list_char = []
    for font in line:
        for char in font:
            list_char.append(char.attrib["c"])
    return "".join(list_char)


def accent_and_badtext_handler(badtext: str) -> str:
    try:
        encoded = str(badtext).encode("cp1252")
        goodtext = encoded.decode("utf-8")
    except:
        accented_string = str(badtext)
        badtext = unidecode.unidecode(accented_string)
        encoded = badtext.encode("cp1252")
        goodtext = encoded.decode("utf-8")

    return goodtext


def detect_number_of_columns(blocks_data: List[Dict]) -> Tuple[int, List[float]]:
    """
    Detect the number of columns in a page layout.

    Args:
        blocks_data: List of dictionaries containing block information with
                    keys: 'block_x0', 'block_x1', 'text'

    Returns:
        Tuple of (number_of_columns, column_borders)
    """
    if not blocks_data:
        return 1, []

    # Extract x-coordinates and text lengths
    x0_coords = []
    x1_coords = []
    text_lengths = []

    for block in blocks_data:
        if block.get('text', '').strip():  # Only consider blocks with text
            x0_coords.append(float(block['block_x0']))
            x1_coords.append(float(block['block_x1']))
            text_lengths.append(len(block['text'].strip()))

    if not x0_coords:
        return 1, []

    # Method 1: Simple distribution analysis (from deprecated code)
    page_width = max(x1_coords) if x1_coords else 0
    if page_width == 0:
        return 1, []

    # Calculate the midpoint of the page
    page_midpoint = page_width / 2

    # Count blocks in left and right halves
    left_blocks = 0
    right_blocks = 0
    left_text_count = 0
    right_text_count = 0
    crossing_text_count = 0
    total_text_count = 0

    for i, x0 in enumerate(x0_coords):
        x1 = x1_coords[i]
        text_len = text_lengths[i]
        total_text_count += text_len

        # Check if block crosses the midpoint
        if x0 < page_midpoint <= x1:
            crossing_text_count += text_len
        elif x1 < page_midpoint:
            left_blocks += 1
            left_text_count += text_len
        elif x0 >= page_midpoint:
            right_blocks += 1
            right_text_count += text_len

    # Calculate crossing ratio (from deprecated code)
    if total_text_count > 0:
        crossing_ratio = crossing_text_count / total_text_count

        # If more than 80% of text crosses the midpoint, it's likely
        # single column
        if crossing_ratio > 0.80:
            return 1, []

    # Method 2: Advanced frequency analysis
    # Find the most common x0 coordinates
    x0_frequency = Counter(x0_coords)
    sorted_x0 = sorted(x0_frequency.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_x0) >= 2:
        # Get the two most common x0 positions
        mod1_x0 = min(sorted_x0[0][0], sorted_x0[1][0])
        mod2_x0 = max(sorted_x0[0][0], sorted_x0[1][0])

        # Find corresponding x1 values for these x0 positions
        mod1_x1_values = [x1 for i, x1 in enumerate(x1_coords) if abs(x0_coords[i] - mod1_x0) < 1]
        mod2_x1_values = [x1 for i, x1 in enumerate(x1_coords) if abs(x0_coords[i] - mod2_x0) < 1]

        if mod1_x1_values and mod2_x1_values:
            mod1_x1 = max(mod1_x1_values)  # Right edge of left column
            # mod2_x1 = max(mod2_x1_values)  # Right edge of right column

            # Calculate border between columns
            column_border = (mod2_x0 + mod1_x1) / 2

            # Check if we have significant content in both columns
            left_content = sum(
                text_lengths[i] for i, x0 in enumerate(x0_coords) if (x0 + x1_coords[i]) / 2 < column_border
            )
            right_content = sum(
                text_lengths[i] for i, x0 in enumerate(x0_coords) if (x0 + x1_coords[i]) / 2 >= column_border
            )

            total_content = left_content + right_content
            if total_content > 0:
                left_ratio = left_content / total_content
                right_ratio = right_content / total_content

                # Check if both columns have significant content (between 20% and 80%)
                if 0.2 <= left_ratio <= 0.8 and 0.2 <= right_ratio <= 0.8:
                    return 2, [column_border]

    # Method 3: Simple threshold-based detection
    min_blocks_per_column = 2
    min_text_ratio = 0.1  # At least 10% of text in each column

    if left_blocks >= min_blocks_per_column and right_blocks >= min_blocks_per_column:
        total_text = left_text_count + right_text_count
        if total_text > 0:
            left_text_ratio = left_text_count / total_text
            right_text_ratio = right_text_count / total_text

            if (
                left_text_ratio >= min_text_ratio
                and right_text_ratio >= min_text_ratio
                and abs(left_text_ratio - right_text_ratio) < 0.6
            ):  # Not too imbalanced
                return 2, [page_midpoint]

    # Default to single column
    return 1, []


def get_column_info(blocks_data: List[Dict]) -> Dict:
    """
    Get comprehensive column layout information.

    Args:
        blocks_data: List of dictionaries containing block information

    Returns:
        Dictionary with column layout information
    """
    num_columns, column_borders = detect_number_of_columns(blocks_data)

    column_info = {
        'ncols': num_columns,
        'column_borders': column_borders,
        'left_column_blocks': 0,
        'right_column_blocks': 0,
        'left_column_text': 0,
        'right_column_text': 0,
    }

    if num_columns == 2 and column_borders:
        column_border = column_borders[0]
        for block in blocks_data:
            if block.get('text', '').strip():
                x0 = float(block['block_x0'])
                x1 = float(block['block_x1'])
                text_len = len(block['text'].strip())
                block_center = (x0 + x1) / 2

                if block_center < column_border:
                    column_info['left_column_blocks'] += 1
                    column_info['left_column_text'] += text_len
                else:
                    column_info['right_column_blocks'] += 1
                    column_info['right_column_text'] += text_len

    return column_info


def is_page_extraction_valid(page_rows: List[Dict]) -> bool:
    if len(page_rows) <= 20:
        return False

    return True


def extract_pdf_text(
    pdf_path: str | Path,
    pdf_type: str,
) -> pd.DataFrame:
    """extracts initial features of a pdf by pymupdf library and
    labels it by headers and paragraphs"""

    if isinstance(pdf_path, str):
        pdf_path = Path(pdf_path)

    proper_text_correct = ""
    proper_text_origin = ""
    char_coor = []
    line_coor = []
    block_coor = []
    font_family_info_proper = []
    font_size_info_proper = []
    color_info_proper = []

    # Collect all data in a list of dictionaries
    data_rows = []

    with fitz.open(pdf_path) as doc:

        for page_number, page in enumerate(doc):
            page_rows = []
            log.debug(f"Processing page {page_number}/{len(doc)}")
            dic = page.get_text("dict")

            height = dic["height"]
            width = dic["width"]
            # Define crop bounds for different PDF types
            crop_bounds = {
                "CFR": (60, width - 60, 60, height - 60),
                "FAA_Advisory_Circulars_Data": (20, 530, 30, 680),
                "Arxiv": (0, width, 0, height),
            }

            # Get crop bounds for the PDF type, default to full page if not
            # specified
            crop_x0, crop_x1, crop_y0, crop_y1 = crop_bounds.get(pdf_type, (0, width, 0, height))

            # crop each page to remove headings and footnotes,
            # crop borders are different in different pdf types
            try:
                # if pdf_type is not defined to define the borders,
                # just skip croping.
                page.set_cropbox(fitz.Rect(crop_x0, crop_y0, crop_x1, crop_y1))
            except Exception:
                pass

            # if there is any table in the page, this function gathers
            # the coordinates of all the tables in the page
            tables, table_coordinates = get_table_location(page)
            # if table coordinate for y1 value is larger than the crop_y1
            # value of the page, the table should be continued to the next page
            table_continued_to_next_page = any(ext[3] > crop_y1 for ext in table_coordinates)

            # Extract table cells firsts
            table_cells = extract_table_cells(tables, page, table_coordinates)

            # Add table cells to data rows
            for cell in table_cells:
                cell['PageNumber'] = page_number
                page_rows.append(cell)

            text = page.get_text("xml")
            root = ET.fromstring(re.sub(r"&#([a-zA-Z0-9]+);?", r"[#\1;]", text))
            page_blocks_data = []

            for j, block in enumerate(root):

                bx0, by0, bx1, by1 = block.attrib["bbox"].split(" ")
                bx0, by0, bx1, by1 = float(bx0), float(by0), float(bx1), float(by1)

                # Check if this block is within a table area
                block_in_table = is_table_content(table_coordinates, bx0, bx1, by0, by1)

                # Skip processing if block is within table area (already processed as cells)
                if block_in_table:
                    continue

                # Collect block data for column detection
                block_text = ""
                for line in block:
                    block_text += return_line_of_text(line) + " "
                page_blocks_data.append({'block_x0': bx0, 'block_x1': bx1, 'text': block_text.strip()})

                for i, line in enumerate(block):
                    next_line_page_number = np.nan

                    linex0, liney0, linex1, liney1 = line.attrib["bbox"].split(" ")

                    list_char_x_perfont = []
                    list_char_y_perfont = []
                    font_family_info = {}
                    font_size_info = {}
                    color_info = {}
                    list_font_family = []
                    list_font_size = []
                    for font in line:
                        list_char_text = []
                        for char in font:
                            list_char_text.append(char.attrib["c"])
                            list_char_x_perfont.append(char.attrib["x"])
                            list_char_y_perfont.append(char.attrib["y"])

                        len_font_char = len(list_char_text)
                        font_family = font.attrib["name"]
                        font_size = font.attrib["size"]

                        list_font_family.append(font_family)
                        list_font_size.append(font_size)

                        if font_family in font_family_info.keys():
                            font_family_info[font_family] = font_family_info[font_family] + len_font_char
                        else:
                            font_family_info[font_family] = len_font_char

                        if font_size in font_size_info.keys():
                            font_size_info[font_size] = font_size_info[font_size] + len_font_char
                        else:
                            font_size_info[font_size] = len_font_char

                        # Extract color from each character in this font
                        for char in font:
                            char_color = char.attrib.get("color", "#000000")  # Default to black if no color attribute
                            if char_color in color_info.keys():
                                color_info[char_color] = color_info[char_color] + 1
                            else:
                                color_info[char_color] = 1

                    # get next line and previous line and current line
                    # origin current line
                    text_line_origin = str(return_line_of_text(block[i]))
                    # correct current line by removing accent and bad text
                    text_line_correct = str(accent_and_badtext_handler(text_line_origin)).strip()

                    # if it is a merged line (parts of current_line are readed
                    # as separated lines, so we merged them in rest of the code),
                    # update the value of origin current line and correct current
                    # line
                    if len(str(proper_text_correct).strip()) != 0:
                        text_line_origin = proper_text_origin
                        text_line_correct = proper_text_correct

                    # gather next_line info if next line is in the same block
                    # not next block
                    if i < len(block) - 1:
                        # if next line is in the same block not next block
                        # if next line is in the same block so next line page
                        # number is same as current line page number and get
                        # values of 1
                        next_line_page_number = 1
                        next_line = return_line_of_text(block[i + 1])

                        (_, _, _, _, _, next_y0) = line_info(block[i + 1])

                        next_line_correct = accent_and_badtext_handler(next_line)

                    else:
                        # gather next_line info if next line is in the next block
                        # if next line is in same page
                        if j < len(root) - 1:
                            next_line_page_number = 1
                            next_line = return_line_of_text(root[j + 1][0])
                            (_, _, _, _, _, next_y0) = line_info(root[j + 1][0])
                            next_line_correct = accent_and_badtext_handler(next_line)
                        else:
                            # if next line is in next page
                            next_line_page_number = 0
                            next_line_correct = next_line = ""

                    # gather prev_line info if prev line is in the same block
                    # not prev block
                    if i > 0:
                        prev_line = return_line_of_text(block[i - 1])
                        prev_line_correct = accent_and_badtext_handler(prev_line)
                    else:
                        # gather prev_line info if prev line is in the prev block
                        if j > 0:
                            # if it's in current page
                            prev_line = return_line_of_text(root[j - 1][-1])
                            prev_line_correct = accent_and_badtext_handler(prev_line)
                        else:
                            # if it's in prev page
                            prev_line_correct = prev_line = ""

                    # coordinates of first char and last char in a text_line
                    coordinate = (
                        list_char_x_perfont[0],
                        list_char_y_perfont[0],
                        list_char_x_perfont[-1],
                        list_char_y_perfont[-1],
                    )
                    current_x0 = coordinate[0]
                    current_x1 = coordinate[2]
                    current_y0 = coordinate[1]
                    current_y1 = coordinate[3]

                    # next line spaceing is caculated by using text_line
                    # character coordinates, to merge lines in the rest of the
                    # code
                    if next_line_page_number in [
                        0,
                        1,
                    ]:
                        if next_line_page_number == 1:  # if in same page
                            if float(next_y0) < float(current_y0):
                                next_line_space = abs(float(next_y0) - float(current_y0))
                            else:
                                next_line_space = abs(float(next_y0) - float(current_y0))
                        else:  # if not in same page
                            next_line_space = (float(height) - float(current_y0)) + float(next_y0)
                    else:
                        # if it's last line
                        next_line_space = 0

                    # merging lines (parts of current_line are readed as
                    # separated lines, so we merged them in rest of the code)
                    # thershold to recognize if current_line, should be
                    # merged with next line
                    if next_line_space < 0.1:

                        # below features are affected in merging:
                        # coordinates of first char and last char in a text_line
                        char_coor.append([current_x0, current_y0, current_x1, current_y1])
                        # coordinates of text_line
                        line_coor.append([linex0, liney0, linex1, liney1])
                        # coordinates of block
                        block_coor.append([bx0, by0, bx1, by1])

                        font_family_info_proper.append(font_family_info)
                        font_size_info_proper.append(font_size_info)
                        color_info_proper.append(color_info)

                        if str(next_line_correct).strip().startswith(str(text_line_correct).strip()):
                            proper_text_correct = next_line_correct
                            proper_text_origin = next_line

                        elif str(text_line_correct).strip().endswith(str(next_line_correct).strip()):
                            proper_text_correct = text_line_correct
                            proper_text_origin = text_line_origin

                        else:
                            proper_text_correct = str(text_line_correct).strip() + " " + str(next_line_correct).strip()
                            proper_text_origin = str(text_line_origin).strip() + " " + str(next_line).strip()

                        continue

                    else:
                        # if merging lines condition is not met
                        proper_text_correct = ""
                        proper_text_origin = ""
                        if len(char_coor) > 0:

                            char_coor.append([current_x0, current_y0, current_x1, current_y1])
                            line_coor.append([linex0, liney0, linex1, liney1])
                            block_coor.append([bx0, by0, bx1, by1])

                            font_family_info_proper.append(font_family_info)
                            font_size_info_proper.append(font_size_info)
                            color_info_proper.append(color_info)

                            current_x0, current_y0 = char_coor[0][:2]
                            current_x1, current_y1 = char_coor[-1][2:]
                            linex0, liney0 = line_coor[0][:2]
                            linex1, liney1 = line_coor[-1][2:]
                            bx0, by0 = block_coor[0][:2]
                            bx1, by1 = block_coor[-1][2:]

                            font_family_info = font_family_info_proper[0]
                            for dict2 in font_family_info_proper[1:]:
                                font_family_info = merge_dict(font_family_info, dict2)

                            font_size_info = font_size_info_proper[0]
                            for dict2 in font_size_info_proper[1:]:
                                font_size_info = merge_dict(font_size_info, dict2)

                            color_info = color_info_proper[0]
                            for dict2 in color_info_proper[1:]:
                                color_info = merge_dict(color_info, dict2)

                            font_family_info_proper = []
                            font_size_info_proper = []
                            color_info_proper = []
                            char_coor = []
                            line_coor = []
                            block_coor = []

                    # detects table per line
                    is_table = is_table_content(table_coordinates, bx0, bx1, by0, by1)

                    # Create row data dictionary
                    row_data = {
                        "origin_text": text_line_origin,
                        "text": text_line_correct,
                        # "correct_prev_line": prev_line_correct,
                        # "correct_next_line": next_line_correct,
                        "x0": float(current_x0),
                        "y0": float(current_y0),
                        "x1": float(current_x1),
                        "y1": float(current_y1),
                        "block_x0": float(bx0),
                        "block_y0": float(by0),
                        "block_x1": float(bx1),
                        "block_y1": float(by1),
                        "line_x0": float(linex0),
                        "line_y0": float(liney0),
                        "line_x1": float(linex1),
                        "line_y1": float(liney1),
                        'PageNumber': page_number,
                        "page_height": float(height),
                        "page_width": float(width),
                        "major_font_family": max(font_family_info, key=font_family_info.get),
                        "major_font_size": max(font_size_info, key=font_size_info.get),
                        "major_color": max(color_info, key=color_info.get) if color_info else "#000000",
                        "font_family_info": font_family_info,
                        "font_size_info": font_size_info,
                        "color_info": color_info,
                        "multiple_font_family": len(font_family_info),
                        "multiple_font_size": len(font_size_info),
                        "table_coordinates": table_coordinates,
                        "horizontal_lines": [],
                        "vertical_lines": [],
                        "table_continued_to_next_page": (table_continued_to_next_page),
                        "is_table": is_table,
                        "crop_x0": float(crop_x0),
                        "crop_x1": float(crop_x1),
                        "crop_y0": float(crop_y0),
                        "crop_y1": float(crop_y1),
                        "version": pdf_path.stem,
                        # Column layout information (will be updated after page processing)
                        "ncols": 1,
                        "column_borders": [],
                        "left_column_blocks": 0,
                        "right_column_blocks": 0,
                        "left_column_text": 0,
                        "right_column_text": 0,
                    }

                    page_rows.append(row_data)

                # After processing all blocks on the page, detect column layout
                if page_blocks_data:
                    column_info = get_column_info(page_blocks_data)
                    # Add column info to all rows from this page
                    for row in page_rows:
                        if row.get('PageNumber') == page_number:
                            row.update(column_info)

            # needs_ocr = not is_page_extraction_valid(page_rows)
            data_rows.extend(page_rows)

        df = pd.DataFrame(data_rows)

        # remove rows where text is nan or empty
        df = df[df['text'].notna()]
        df = df[df['text'] != '']
        df = df.reset_index(drop=True)
        df['pdf_idx'] = df.index
        return df


def get_table_location(
    page: fitz.Page,
) -> Tuple[List[TableFinder] | None, List[Tuple[float, float, float, float]]]:
    """Enhanced table detection combining PyMuPDF built-in and custom methods"""

    # Try PyMuPDF's built-in detection first
    try:
        builtin_tables = page.find_tables(
            # strategy="lines_strict", # this works, but it makes each cell a single pdf_line
        )
        if builtin_tables:
            # Convert to your expected format
            table_rects = [table.bbox for table in builtin_tables]
            return builtin_tables, table_rects
    except Exception as e:
        log.warning(f"Built-in table detection failed: {e}")
    return None, []


def extract_table_cells(
    tables: List[TableFinder], page: fitz.Page, table_coordinates: List[Tuple[float, float, float, float]]
) -> List[Dict]:
    """
    Extract individual table cells from detected tables.

    Parameters
    ----------
    page: fitz.Page
        The PDF page object
    table_coordinates: List[Tuple[float, float, float, float]]
        List of table bounding boxes

    Returns
    -------
    List[Dict]
        List of cell data dictionaries
    """
    if not tables:
        return []

    cell_data = []
    cell_bboxs = []
    # Get tables using PyMuPDF's built-in detection
    for table_idx, table in enumerate(tables):
        # Extract table structure
        table_text = table.extract()

        # Get table bounding box
        table_bbox = table.bbox
        # Process each cell in the table
        for row_idx, row in enumerate(table.rows):
            for col_idx, cell_bbox in enumerate(row.cells):
                # Skip if cell_bbox is None or invalid
                if cell_bbox is None or len(cell_bbox) != 4:
                    continue

                # if cell overlaps with any other cell, skip it
                if any(
                    calculate_iou(cell_bbox, bbox) > 0.1 for bbox in cell_bboxs if bbox is not None and len(bbox) == 4
                ):
                    continue
                cell_content = table_text[row_idx][col_idx]
                if "january" in cell_content.lower():
                    log.debug("Cell bbox: %s", cell_bbox)
                # remove content from table text
                table_text[row_idx][col_idx] = ""
                if cell_content and cell_content.strip():  # Only process non-empty cells
                    # Create cell data dictionary
                    cell_dict = {
                        "origin_text": cell_content,
                        "text": accent_and_badtext_handler(cell_content).strip(),
                        "x0": float(cell_bbox[0]),
                        "y0": float(cell_bbox[1]),
                        "x1": float(cell_bbox[2]),
                        "y1": float(cell_bbox[3]),
                        "block_x0": float(cell_bbox[0]),
                        "block_y0": float(cell_bbox[1]),
                        "block_x1": float(cell_bbox[2]),
                        "block_y1": float(cell_bbox[3]),
                        "line_x0": float(cell_bbox[0]),
                        "line_y0": float(cell_bbox[1]),
                        "line_x1": float(cell_bbox[2]),
                        "line_y1": float(cell_bbox[3]),
                        'PageNumber': page.number,
                        "page_height": float(page.rect.height),
                        "page_width": float(page.rect.width),
                        "major_font_family": "table_cell",  # Placeholder
                        "major_font_size": "12",  # Placeholder
                        "major_color": "#000000",  # Placeholder
                        "font_family_info": {"table_cell": len(cell_content)},
                        "font_size_info": {"12": len(cell_content)},
                        "color_info": {"#000000": len(cell_content)},
                        "multiple_font_family": 1,
                        "multiple_font_size": 1,
                        "table_coordinates": [table_bbox],
                        "horizontal_lines": [],
                        "vertical_lines": [],
                        "table_continued_to_next_page": False,
                        "is_table": True,
                        "crop_x0": 0.0,
                        "crop_x1": float(page.rect.width),
                        "crop_y0": 0.0,
                        "crop_y1": float(page.rect.height),
                        "version": "table_cell",
                        "ncols": 1,
                        "column_borders": [],
                        "left_column_blocks": 0,
                        "right_column_blocks": 0,
                        "left_column_text": 0,
                        "right_column_text": 0,
                        "table_row": row_idx,
                        "table_col": col_idx,
                        "table_id": table_idx,
                    }
                    cell_bboxs.append(cell_bbox)
                    cell_data.append(cell_dict)

    return cell_data


def calculate_cell_bbox(table_bbox, row_idx, col_idx, num_rows, num_cols):
    """
    Calculate approximate cell bounding box based on table structure.

    Parameters
    ----------
    table_bbox: fitz.Rect
        Table bounding box
    row_idx: int
        Row index
    col_idx: int
        Column index
    num_rows: int
        Total number of rows
    num_cols: int
        Total number of columns

    Returns
    -------
    Tuple[float, float, float, float]
        Cell bounding box (x0, y0, x1, y1)
    """
    x0, y0, x1, y1 = table_bbox

    # Calculate cell dimensions
    cell_width = (x1 - x0) / num_cols
    cell_height = (y1 - y0) / num_rows

    # Calculate cell coordinates
    cell_x0 = x0 + (col_idx * cell_width)
    cell_y0 = y0 + (row_idx * cell_height)
    cell_x1 = cell_x0 + cell_width
    cell_y1 = cell_y0 + cell_height

    return (cell_x0, cell_y0, cell_x1, cell_y1)


# Example usage
if __name__ == "__main__":
    from ai_doc_parser import CFR_PDF, EASA_DIR, LATEX_PDF

    logging.basicConfig(level=logging.DEBUG)

    input_path = (
        EASA_DIR
        / "Easy Access Rules for Air Traffic Controllers_ Licensing and Certification _Regulation _EU_ 2015_340_ _PDF_.pdf"
    )
    output_dir = input_path.parent / "pdf_extracted"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.csv"
    df = extract_pdf_text(input_path, "")
    df.to_csv(output_path, index=False)
    log.info("Saved to %s", output_path)
