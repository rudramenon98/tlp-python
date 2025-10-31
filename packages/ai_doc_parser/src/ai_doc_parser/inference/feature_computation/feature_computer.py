import logging
import re
from ast import Dict
from enum import Enum
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
from ai_doc_parser.inference.feature_computation.identify_headers_and_footers import (
    identify_headers_and_footnotes,
)
from ai_doc_parser.inference.feature_computation.order_blocks import order_blocks
from ai_doc_parser.text_class import TextClass

log = logging.getLogger(__name__)


def bin_feature(df: pd.DataFrame, feature: str) -> pd.Series:
    # split into 9 bins based on mean and standard deviation
    # Coerce to numeric and work with finite values for stats
    feature_series = pd.to_numeric(df[feature], errors="coerce")
    feature_values = feature_series.values
    finite_mask = np.isfinite(feature_values)
    finite_values = feature_values[finite_mask]

    # If no finite values, return middle bin
    if finite_values.size == 0:
        return pd.Series([4] * len(df), index=df.index)

    mean_val = float(np.mean(finite_values))
    std_val = float(np.std(finite_values))

    # Build bin edges (10 edges -> 9 bins)
    if std_val < 1e-10:
        # Low-variance: use quantile or linear edges
        min_val = float(np.min(finite_values))
        max_val = float(np.max(finite_values))
        if min_val == max_val:
            return pd.Series([4] * len(df), index=df.index)
        # Use quantiles to better reflect data distribution
        edges = np.quantile(finite_values, np.linspace(0.0, 1.0, num=10))
    else:
        # Mean ± std multiples
        edges = np.array(
            [
                -np.inf,
                mean_val - 2.0 * std_val,
                mean_val - 1.0 * std_val,
                mean_val - 0.5 * std_val,
                mean_val - 0.125 * std_val,
                mean_val + 0.125 * std_val,
                mean_val + 0.5 * std_val,
                mean_val + 1.0 * std_val,
                mean_val + 2.0 * std_val,
                np.inf,
            ],
            dtype=float,
        )

    # Ensure edges length is 10; if produced 10 finite edges, extend to cover tails
    if edges.size == 10:
        # Replace first/last with -inf/inf to cover outliers
        edges[0] = -np.inf
        edges[-1] = np.inf
    elif edges.size > 10:
        # Reduce to 10 by selecting evenly spaced edges
        idx = np.linspace(0, edges.size - 1, num=10).round().astype(int)
        edges = edges[idx]
        edges[0] = -np.inf
        edges[-1] = np.inf
    else:
        # Increase to 10 using linspace between min/max
        finite_min = float(np.min(finite_values))
        finite_max = float(np.max(finite_values))
        edges = np.linspace(finite_min, finite_max, num=10)
        edges[0] = -np.inf
        edges[-1] = np.inf

    # Make strictly increasing by enforcing minimal epsilon on ties
    eps = 1e-9
    for i in range(1, edges.size):
        if not np.isfinite(edges[i - 1]):
            continue
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps

    # Perform cut with dynamic number of bins; return integer codes starting at 0
    binned_codes = pd.cut(
        feature_series,
        bins=edges,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )

    # Normalize codes to start at 0 and cap at 8
    binned_series = binned_codes.astype("Float64")  # preserve NaN
    if not binned_series.isna().all():
        # Determine the bin index where the mean value falls (pre-normalization)
        mean_code_raw = int(
            pd.cut(
                pd.Series([mean_val]),
                bins=edges,
                labels=False,
                include_lowest=True,
                duplicates="drop",
            ).iloc[0]
        )

        # Normalize sample codes and the mean-code the same way
        min_code = int(binned_series.min(skipna=True))
        binned_series = (binned_series - min_code).clip(lower=0, upper=8)
        mean_code_norm = int(max(0, min(8, mean_code_raw - min_code)))

        # Fill NaNs with the mean bin (after normalization)
        binned_series = binned_series.fillna(mean_code_norm)

    return binned_series


#### Line Spacing Features ####
def left_indent(row: pd.Series) -> float:
    """Calculate left indent based on column layout."""
    cols = row["ncols"]
    line_x0 = row["line_x0"]

    if cols == 1:
        return line_x0
    if cols == 2:
        block_x0 = row["block_x0"]
        left_indent = line_x0 - block_x0
        return left_indent
    return 0.0


def right_space(row: pd.Series) -> float:
    """Calculate right space based on column layout."""
    cols = row["ncols"]
    width = row["page_width"]
    line_x1 = row["line_x1"]
    block_x1 = row["block_x1"]

    if int(cols) == 1:
        right_space = width - line_x1
        return right_space
    if int(cols) == 2:
        right_space = block_x1 - line_x1
        return right_space
    return 0.0


def space_above(row: pd.Series, prev_row: pd.Series, next_row: pd.Series) -> float:
    """Get previous line space."""
    if prev_row is None:
        return np.nan
    return float(row["line_y0"]) - float(prev_row["line_y0"])


def space_below(row: pd.Series, prev_row: pd.Series, next_row: pd.Series) -> float:
    """Get next line space."""
    return float(row["line_y0"]) - float(next_row["line_y0"])


def more_space_below(row: pd.Series) -> int:
    """Check if there is more space below than above."""
    return get_ternary_diff(row["space_below"], row["space_above"])


#### Font Features ####
def font_size(row: pd.Series) -> float:
    return float(row["major_font_size"])


def is_bold(row: pd.Series) -> int:
    major_font_family = row["major_font_family"]
    return int("bold" in str(major_font_family).lower())


def is_italic(row: pd.Series) -> int:
    major_font_family = row["major_font_family"]
    return int("italic" in str(major_font_family).lower())


def font_color(row: pd.Series) -> str:
    return row["major_color"]


#### Text Features ####
def num_chars(row: pd.Series) -> int:
    """Count number of characters in text."""
    text = str(row["text"].strip())
    return len(text)


def num_words(row: pd.Series) -> int:
    """Count total number of words in text."""
    text = row["text"]
    count = len(text.split())
    return count


def is_all_caps(row: pd.Series) -> int:
    """Check if text is all uppercase."""
    text = str(row["text"])
    return int(text.isupper())


def is_title_case(row: pd.Series) -> int:
    """Check if text is title case."""
    text = str(row["text"])
    if text.istitle():
        return 1
    # remove all non-alpha words
    words = text.split()
    words = [word for word in words if word.isalpha()]
    if len(words) == 0:
        return 0
    num_capitlized = sum(1 for word in words if word[0].isupper())
    percent_capitalized = num_capitlized / len(words)
    return int(percent_capitalized > 0.5)


def num_of_dots(row: pd.Series) -> int:
    """Count total number of dots in text."""
    text = row["text"]
    if first_word_compound(row):
        # exclude first word from text
        text = text.split()[1:]
        text = " ".join(text)
    counts = text[1:].count(".")
    return counts


def num_of_spaces(row: pd.Series) -> int:
    """Count number of spaces in text."""
    text = str(row["text"].strip())
    return text.count(" ")


def max_contiguous_dots(row: pd.Series) -> int:
    """Get maximum number of contiguous dots in text. Returns the length of the longest contiguous sequence of dots."""
    text = str(row["text"].strip())
    text = text.replace(" ", "")
    matches = re.findall(r"\.+", text)
    return max((len(m) for m in matches), default=0)


def max_contiguous_spaces(row: pd.Series) -> int:
    """Get maximum number of contiguous spaces in text. Returns the length of the longest contiguous sequence of spaces."""
    text = str(row["text"].strip())
    text = text.replace(".", "")
    matches = re.findall(r" +", text)
    return max((len(m) for m in matches), default=0)


#### Text Starting Words Features ####
def first_char_isdigit(row: pd.Series) -> int:
    """Check if first character is a digit."""
    text = row["text"]
    return int(text and text[0].isdigit())


def first_char_bullet(row: pd.Series) -> int:
    """Check if first character is a special symbol."""
    text = row["text"]
    symbols = "·○◌●◦◆◇◈■□★☆✓▶+-❖➢➣•[#x2014;]§†*¶"
    first_char = text[0]
    if first_char in symbols:
        return 1
    return 0


def first_char_upper(row: pd.Series) -> int:
    """Check if first character is uppercase."""
    line = row["text"]
    return int(line[0].isupper())


def first_word_compound(row: pd.Series) -> int:
    """Check if first word is a compound word.
    Compound words follow the following rules:
    - Has more than one character
    - Has at least one non-alpha character
    - Is not all digits
    """
    text = str(row["text"])
    first_word = text.split(r"[\s\t]+")[0].strip()
    if len(first_word) <= 1:
        return 0
    if not any(char.isalpha() for char in first_word):
        return 0
    if all(char.isdigit() for char in first_word):
        return 0
    return 1


def starts_with_keyword(row: pd.Series) -> int:
    """Check if text starts with special words like appendix, reference, etc."""
    text = row["text"]
    special_words = ["appendix", "reference", "chapter", "section", "subsection"]
    is_special_word = bool(list(filter(text.lower().startswith, special_words)))
    is_special_word_in_first_word = any(
        ext in text.lower().split()[0] for ext in special_words
    )
    return int(is_special_word or is_special_word_in_first_word)


def first_char_special(row: pd.Series) -> int:
    """Check if text starts with a special character."""
    text = row["text"]
    punctuation = """'"\\<>/@#$%^&*~"""
    return int(text.startswith(tuple(punctuation)))


#### Text Ending Words Features ####
def last_char_upper(row: pd.Series) -> int:
    """Check if text ends with an uppercase letter."""
    text = str(row["text"].strip())
    return int(text[-1].isupper())


def last_char_punctuation(row: pd.Series) -> int:
    """Check if text ends with a punctuation character."""
    text = str(row["text"].strip())
    return int(text[-1] in "()[]{};:,")


def last_word_digit(row: pd.Series) -> int:
    """Check if text ends with a digit."""
    text = str(row["text"].strip())
    return int(text[-1].isdigit())


def last_word_roman(row: pd.Series) -> int:
    """Check if text ends with a Roman numeral."""
    text = str(row["text"].strip())
    last_word = str(text.strip().split(" ")[-1])
    pattern = re.compile(
        "^M{0,3}" "(CM|CD|D?C{0,3})?" "(XC|XL|L?X{0,3})?" "(IX|IV|V?I{0,3})?$",
        re.VERBOSE,
    )
    match = re.match(pattern, last_word)
    return int(match is not None)


def prev_page_diff(row: pd.Series, prev_row: pd.Series, next_row: pd.Series) -> float:
    """Get difference in page number between previous and current row."""
    if prev_row is None:
        return np.nan
    return int(row["PageNumber"] != prev_row["PageNumber"])


def get_ternary_diff(row: pd.Series, row2: pd.Series, feature: str) -> float:
    """Get difference between two values as a ternary value."""
    if row is None and row2 is None:
        return 0
    if row is None or row2 is None:
        return 1
    val1 = row[feature]
    val2 = row2[feature]
    if val1 is None and val2 is None:
        return 0
    if val1 is None or val2 is None:
        return 1
    if val1 == val2:
        return 0
    if val1 > val2:
        return 1
    return -1


def get_diff_vectorized(
    df: pd.DataFrame, df_shifted: pd.DataFrame, feature: str
) -> pd.Series:
    """Vectorized version of get_diff for entire DataFrame."""
    return df[feature] - df_shifted[feature]


def get_binary_diff_vectorized(
    df: pd.DataFrame, df_shifted: pd.DataFrame, feature: str
) -> pd.Series:
    """Vectorized version of get_binary_diff for entire DataFrame."""
    return (df[feature] != df_shifted[feature]).astype(int)


def get_ternary_diff_vectorized(
    df: pd.DataFrame,
    df_shifted: pd.DataFrame,
    feature: str,
    diff_threshold: float = 0.0,
) -> pd.Series:
    """Vectorized version of get_ternary_diff for entire DataFrame."""
    # Handle None/NaN values
    mask_both_none = df[feature].isna() & df_shifted[feature].isna()
    mask_either_none = df[feature].isna() | df_shifted[feature].isna()

    # Initialize result with NaN
    result = pd.Series(np.nan, index=df.index)

    # Both None -> 0
    result[mask_both_none] = 0

    # Either None -> 1
    result[mask_either_none & ~mask_both_none] = 1

    # Neither None - compare values
    mask_neither_none = ~mask_either_none
    if mask_neither_none.any():
        val1 = df.loc[mask_neither_none, feature]
        val2 = df_shifted.loc[mask_neither_none, feature]
        # Equal -> 0
        equal_mask = val1 == val2
        result.loc[mask_neither_none & equal_mask] = 0

        # Greater -> 1, Less -> -1
        greater_mask = val1 > (val2 + diff_threshold)
        result.loc[mask_neither_none & greater_mask] = 1
        less_mask = val1 < (val2 - diff_threshold)
        result.loc[mask_neither_none & less_mask] = -1
        result.loc[mask_neither_none & ~greater_mask & ~less_mask & ~equal_mask] = 0

    return result


def prev_line_space_below_diff(
    row: pd.Series, prev_row: pd.Series, next_row: pd.Series
) -> float:
    """Get difference in space below between previous and current row."""
    if row["space_above"] == row["space_below"]:
        return 0
    if row["space_above"] > row["space_below"]:
        return 1
    return -1


def prev_line_left_margin_diff(
    row: pd.Series, prev_row: pd.Series, next_row: pd.Series
) -> float:
    """Get difference in left margin between previous and current row."""
    return get_ternary_diff(row, prev_row, "left_indent")


def prev_line_right_space_diff(
    row: pd.Series, prev_row: pd.Series, next_row: pd.Series
) -> float:
    """Get difference in right space between previous and current row."""
    return get_ternary_diff(row, prev_row, "right_space")


def prev_line_font_size_diff(
    row: pd.Series, prev_row: pd.Series, next_row: pd.Series
) -> float:
    """Get difference in font size between previous and current row."""
    return get_ternary_diff(row, prev_row, "font_size")


def prev_line_font_family_diff(
    row: pd.Series, prev_row: pd.Series, next_row: pd.Series
) -> float:
    """Get difference in font family between previous and current row."""
    return get_ternary_diff(row, prev_row, "major_font_family")


def next_page_diff(row: pd.Series, prev_row: pd.Series, next_row: pd.Series) -> float:
    """Get difference in page number between current and next row."""
    return int(row["PageNumber"] != next_row["PageNumber"])


def next_line_left_margin_diff(
    row: pd.Series, prev_row: pd.Series, next_row: pd.Series
) -> float:
    """Get difference in left margin between current and next row."""
    return get_ternary_diff(row, next_row, "left_indent")


def next_line_right_space_diff(
    row: pd.Series, prev_row: pd.Series, next_row: pd.Series
) -> float:
    """Get difference in right space between current and next row."""
    return get_ternary_diff(row, next_row, "right_space")


def next_line_font_size_diff(
    row: pd.Series, prev_row: pd.Series, next_row: pd.Series
) -> float:
    """Get difference in font size between current and next row."""
    return get_ternary_diff(row, next_row, "font_size")


def next_line_font_family_diff(
    row: pd.Series, prev_row: pd.Series, next_row: pd.Series
) -> float:
    """Get difference in font family between current and next row."""
    return get_ternary_diff(row, next_row, "major_font_family")


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute spacing and text features for the DataFrame.

    Args:
        df: Input DataFrame containing text and layout information

    Returns:
        DataFrame with computed features added as new columns
    """
    log.info("Computing features")
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Clean text data
    df["text"] = df["text"].apply(lambda x: str(x).rstrip())

    # Drop rows with no text
    df = df[df["text"] != ""]

    # Add ID column
    df["ID"] = [i for i in range(len(df))]

    # Identify headers and footnotes
    log.info("Identifying headers and footnotes")
    df = identify_headers_and_footnotes(df)

    # Get class
    def get_class(row: pd.Series) -> TextClass | None:
        is_table = row["is_table"]
        is_header = row["is_header"]
        is_footnote = row["is_footnote"]
        if is_table:
            return TextClass.TABLE
        if is_header:
            return TextClass.HEADER
        if is_footnote:
            return TextClass.FOOT_NOTE
        return None

    df["ExtractedClass"] = df.apply(get_class, axis=1)
    df["ExtractedClassName"] = df["ExtractedClass"].apply(
        lambda x: TextClass(x).name if not (x is None or pd.isna(x)) else None
    )

    # Order blocks by their position
    log.info("Ordering blocks")
    df = order_blocks(df)

    # go page by page and compute which column each row is in
    for page_number, page_df in df.groupby("PageNumber"):
        ncols = max(page_df["ncols"])
        if ncols == 1:
            # Update col_num for single-column pages
            # Use boolean indexing to ensure we update the correct rows
            mask = df["PageNumber"] == page_number
            df.loc[mask, "col_num"] = 0
            continue
        x_range = (page_df["block_x0"].min(), page_df["block_x0"].max())
        width = x_range[1] - x_range[0]
        col_width = width / ncols
        col_lines = [col_width * i + x_range[0] for i in range(ncols + 1)]
        for row_index, row in page_df.iterrows():
            x0 = row["block_x0"]
            for i in range(len(col_lines)):
                if x0 <= col_lines[i]:
                    df.loc[row_index, "col_num"] = i
                    break

    # Define dictionary mapping column names to their corresponding functions
    features_to_bin = [
        "left_indent",
        "right_space",
        "space_above",
        "space_below",
        "font_size",
        "num_chars",
        "num_words",
        "num_of_dots",
        "num_of_spaces",
        "max_contiguous_dots",
        "max_contiguous_spaces",
    ]
    feature_functions: Dict[str, Callable[[pd.Series], float]] = {
        # Line Spacing Features
        "left_indent": left_indent,
        "right_space": right_space,
        # Font Features
        "font_size": font_size,
        "is_bold": is_bold,
        "is_italic": is_italic,
        "font_color": font_color,
        # Text Features
        "num_chars": num_chars,
        "num_words": num_words,
        "is_all_caps": is_all_caps,
        "is_title_case": is_title_case,
        "num_of_dots": num_of_dots,
        "num_of_spaces": num_of_spaces,
        "max_contiguous_dots": max_contiguous_dots,
        "max_contiguous_spaces": max_contiguous_spaces,
        # Text Starting Words Features
        "first_char_isdigit": first_char_isdigit,
        "first_char_bullet": first_char_bullet,
        "first_char_upper": first_char_upper,
        "first_word_compound": first_word_compound,
        "starts_with_keyword": starts_with_keyword,
        "first_char_special": first_char_special,
        # Text Ending Words Features
        "last_char_upper": last_char_upper,
        "last_char_punctuation": last_char_punctuation,
        "last_word_digit": last_word_digit,
        "last_word_roman": last_word_roman,
    }

    class DF_SHIFT(Enum):
        CURR = 0
        PREV = 1
        NEXT = 2

    df["original_index"] = range(len(df))
    extracted_class_df = df[df["ExtractedClass"].notna()]
    df = df[df["ExtractedClass"].isna()]

    # Apply each function to create the corresponding column
    for column_name, func in feature_functions.items():
        log.info(f"Computing {column_name}")
        df[column_name] = df.apply(func, axis=1)

    # Compute relative features using vectorized operations
    log.info("Computing relative features")

    df["space_above"] = get_diff_vectorized(df, df.shift(1), "line_y0")
    df["space_below"] = get_diff_vectorized(df.shift(-1), df, "line_y0")
    df.loc[df["space_above"] <= 0, "space_above"] = np.nan
    df.loc[df["space_below"] <= 0, "space_below"] = np.nan

    # Compute changed features
    df["bold_changed"] = get_binary_diff_vectorized(df, df.shift(1), "is_bold")
    df["italic_changed"] = get_binary_diff_vectorized(df, df.shift(1), "is_italic")
    df["font_size_changed"] = get_ternary_diff_vectorized(df, df.shift(1), "font_size")
    df["font_color_changed"] = get_binary_diff_vectorized(df, df.shift(1), "font_color")
    df["font_family_changed"] = get_binary_diff_vectorized(
        df, df.shift(1), "major_font_family"
    )

    df["prev_page_diff"] = get_binary_diff_vectorized(df, df.shift(1), "PageNumber")
    df["prev_line_space_below_diff"] = get_ternary_diff_vectorized(
        df, df.shift(1), "space_below"
    )
    df["prev_line_left_margin_diff"] = get_ternary_diff_vectorized(
        df, df.shift(1), "left_indent"
    )
    df["prev_line_right_space_diff"] = get_ternary_diff_vectorized(
        df, df.shift(1), "right_space"
    )
    df["prev_line_font_size_diff"] = get_ternary_diff_vectorized(
        df, df.shift(1), "font_size"
    )
    df["prev_line_font_family_diff"] = get_binary_diff_vectorized(
        df, df.shift(1), "major_font_family"
    )
    df["next_page_diff"] = get_binary_diff_vectorized(df, df.shift(-1), "PageNumber")
    df["next_line_font_family_diff"] = get_binary_diff_vectorized(
        df, df.shift(-1), "major_font_family"
    )

    prev_page_diff = df[df["prev_page_diff"] == 1].index
    next_page_diff = df[df["next_page_diff"] == 1].index

    prev_line_features: List[str] = [
        "prev_line_space_below_diff",
        "prev_line_left_margin_diff",
        "prev_line_right_space_diff",
        "prev_line_font_size_diff",
        "prev_line_font_family_diff",
        "space_above",
    ]
    next_line_features: List[str] = [
        "next_line_left_margin_diff",
        "next_line_right_space_diff",
        "next_line_font_size_diff",
        "next_line_font_family_diff",
        "space_below",
    ]
    for feature in prev_line_features:
        df.loc[prev_page_diff, feature] = np.nan
    for feature in next_line_features:
        df.loc[next_page_diff, feature] = np.nan

    # bin certain features
    log.info("Binning features")
    for feature in features_to_bin:
        bin_feature_name = f"{feature}_binned"
        df[bin_feature_name] = bin_feature(df, feature)
    df["pdf_idx"] = df.index

    # doing these after binning to avoid rounding errors
    # I just want to know if space below is in the same bin as the previous row
    df["more_space_below"] = 0
    df.loc[df["space_below"] > df["space_above"] + 1.5, "more_space_below"] = 1
    df.loc[df["space_below"] < df["space_above"] - 1.5, "more_space_below"] = -1
    df["next_line_left_margin_diff"] = get_ternary_diff_vectorized(
        df, df.shift(-1), "left_indent_binned"
    )
    df["next_line_right_space_diff"] = get_ternary_diff_vectorized(
        df, df.shift(-1), "right_space_binned"
    )
    df["next_line_font_size_diff"] = get_ternary_diff_vectorized(
        df, df.shift(-1), "font_size_binned"
    )

    # add back the extracted class df and sort by original_index
    df = pd.concat([df, extracted_class_df])
    df = df.sort_values(by="original_index")
    df = df.drop(columns=["original_index"])
    return df


if __name__ == "__main__":
    from ai_doc_parser import CFR_PDF as pdf_path

    logging.basicConfig(level=logging.DEBUG)

    pdf_path = Path(
        "C:/Users/r123m/Documents/enginius/source/ai-pdf-parser/data/documents/validation/B2LILNN15.pdf"
    )

    df_path = pdf_path.parent / "pdf_extracted" / f"{pdf_path.stem}.csv"
    output_path = df_path.parent.parent / "computed_features"
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{df_path.stem}.csv"

    df = pd.read_csv(df_path)
    df = df.head(1000)
    features = compute_features(df)
    features.to_csv(output_file, index=False)
    log.info("Saved to %s", output_file)

    # df_path = pdf_path.parent / "pdf_labelled" / f"{pdf_path.stem}.csv"
    # output_path = df_path
    # output_path.mkdir(parents=True, exist_ok=True)
    # output_file = output_path / f"{df_path.stem}.csv"

    # df = pd.read_csv(df_path)
    # features = compute_features(df)
    # features.to_csv(output_file, index=False)
    # print(f"Saved to {output_file}")
