from typing import Any, Dict, List

import pandas as pd

from ai_doc_parser.text_class import AI_PARSED_CLASSES, CONTINUE_PAIRS, TextClass


def _apply_heuristics(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply rule-based corrections to AI predictions

    Args:
        elements: List of elements from AI service

    Returns:
        List of corrected elements
    """
    if not elements:
        return elements

    # Convert to list for easier manipulation
    corrected_elements = elements.copy()
    match = '.....'

    # Rule 1: Identify table of contents entries
    for i, element in enumerate(corrected_elements):
        if match in element['text']:
            corrected_elements[i]['type'] = self.TOC

    # Rule 2: Context-based corrections
    for i in range(len(corrected_elements)):
        current_type = corrected_elements[i]['type']

        # First element: if type 3, make it a heading
        if i == 0:
            if current_type == 3:
                corrected_elements[i]['type'] = self.HEADING
            continue

        # Last element: context check
        if i == len(corrected_elements) - 1:
            if i > 0 and corrected_elements[i - 1]['type'] not in [self.HEADING, 3] and current_type == 3:
                corrected_elements[i]['type'] = self.HEADING
            continue

        # Middle elements: apply context rules
        prev_type = corrected_elements[i - 1]['type']
        next_type = corrected_elements[i + 1]['type']

        # TOC context rule
        if prev_type == self.TOC and next_type == self.TOC and current_type != self.HEADING:
            corrected_elements[i]['type'] = self.TOC

        # Heading context rule
        if prev_type not in [self.HEADING, 3] and current_type == 3:
            corrected_elements[i]['type'] = self.HEADING

        # Paragraph context rules
        if prev_type == self.HEADING and current_type == 1:
            if next_type == 1:
                corrected_elements[i]['type'] = self.PARAGRAPH
            elif next_type == 0:
                corrected_elements[i]['type'] = 3  # Special case

        if prev_type == 3 and current_type == 1:
            if next_type == 1:
                corrected_elements[i]['type'] = self.PARAGRAPH
            elif next_type == 0:
                corrected_elements[i]['type'] = 3  # Special case

    return corrected_elements


def post_classification_heuristics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply rule-based corrections to AI predictions in a DataFrame

    Args:
        df: DataFrame with 'text' and 'PredictedClass' columns

    Returns:
        DataFrame with corrected predictions
    """
    if df.empty:
        return df

    # remove rows where the 'FinalClass' is not in AI_PARSED_CLASSES.
    # these will be added back later
    df['original_index'] = df.index

    # Create a copy to avoid modifying the original DataFrame
    corrected_df = df.copy()
    corrected_df = corrected_df[corrected_df['PredictedClass'].isin(AI_PARSED_CLASSES)]

    # Heuristcs for headings
    known_heading_text = ["table of contents", "list of figures", "list of tables", "contents"]
    for i, row in corrected_df.iterrows():
        if any(text == row['text'].lower() for text in known_heading_text):
            corrected_df.loc[i, 'FinalClass'] = TextClass.HEADING



    ##### TOC RULES #####
    # If num of consecutive '.' > 5, then it is a table of contents
    toc_pattern = r'\.{5,}'  # Match 5 or more literal dots
    toc_mask = corrected_df['text'].str.replace(' ', '').str.contains(toc_pattern)
    corrected_df.loc[toc_mask, 'FinalClass'] = TextClass.TOC
    
    def starts_and_ends_with_digit(df: pd.DataFrame) -> pd.Series:
        """Check if text starts and ends with digits for each row in the DataFrame"""

        def check_single_row(text):
            words = str(text).split()
            if len(words) < 3:
                return False
            first_word = words[0]
            last_word = words[-1]
            return first_word.isdigit() and last_word.isdigit()

        return df['text'].apply(check_single_row)

    corrected_df.loc[starts_and_ends_with_digit(corrected_df), 'FinalClass'] = TextClass.TOC

    # Get previous and next line classes for vectorized operations
    prev_line = corrected_df['FinalClass'].shift(1)
    next_line = corrected_df['FinalClass'].shift(-1)
    
    # Define enum_regex pattern to match various enumeration formats
    enum_regex = r'^(\d+\.\s|\(\d+\)\s|\([a-z]\)\s|[a-z]\.\s)'

    # If previous line and next line are TOC, and current is not a heading, then it is TOC
    lone_toc_mask = ~corrected_df['FinalClass'].isin([TextClass.TOC, TextClass.HEADING])
    corrected_df.loc[
        lone_toc_mask
        & (prev_line == TextClass.TOC)
        & (next_line == TextClass.TOC)
        & (corrected_df['FinalClass'] != TextClass.HEADING),
        'FinalClass',
    ] = TextClass.TOC


    # If previous two lines are not TOC and next two lines are not TOC, then current line
    # should not be TOC
    # TODO

    ##### HEADING RULES #####
    # If currenty line is HEADING_CONT, but it has different font size than the previous line, then it should be HEADING
    corrected_df.loc[
        (corrected_df['FinalClass'] == TextClass.HEADING_CONT)
        & (corrected_df['font_size_binned'] != corrected_df['font_size_binned'].shift(1)),
        'FinalClass',
    ] = TextClass.HEADING
    
    # If previous line is HEADING and next line is HEADING_CONT, and current line is not HEADING, then current line must be HEADING_CONT
    corrected_df.loc[(prev_line == TextClass.HEADING) & 
                     (corrected_df['FinalClass'] != TextClass.HEADING) &
                     (next_line == TextClass.HEADING_CONT), 'FinalClass'] = (
        TextClass.HEADING_CONT
    )

    # If first line in a document page is HEADING_CONT, then it should be HEADING
    is_first_line = corrected_df['PageNumber'] != corrected_df['PageNumber'].shift(-1)
    corrected_df.loc[is_first_line & (corrected_df['FinalClass'] == TextClass.HEADING_CONT), 'FinalClass'] = (
        TextClass.HEADING
    )

    # If previous is HEADING, next one is PARAGRAPH_CONT and current line PARAGRAPH_CONT, set current to PARAGRAPH_CONT
    corrected_df.loc[
        (prev_line == TextClass.HEADING)
        & (next_line == TextClass.PARAGRAPH_CONT)
        & (corrected_df['FinalClass'] == TextClass.PARAGRAPH_CONT),
        'FinalClass',
    ] = TextClass.PARAGRAPH_CONT

    # If previous is HEADING_CONT, current is PARAGRAPH and next is PARAGRAPH_CONT, set current to HEADING_CONT
    corrected_df.loc[
        (prev_line == TextClass.HEADING_CONT)
        & (next_line == TextClass.PARAGRAPH)
        & (corrected_df['FinalClass'] == TextClass.PARAGRAPH_CONT),
        'FinalClass',
    ] = TextClass.HEADING_CONT

    # If previous is HEADING_CONT, current is PARAGRAPH_CONT, next is PARAGRAPH_CONT, then current is PARAGRAPH
    corrected_df.loc[
        (prev_line == TextClass.HEADING_CONT)
        & (next_line == TextClass.PARAGRAPH_CONT)
        & (corrected_df['FinalClass'] == TextClass.PARAGRAPH_CONT),
        'FinalClass',
    ] = TextClass.PARAGRAPH

    # If line starts with [#x2014;], it is a BULLET_LIST
    bullet_mask = corrected_df['text'].str.strip().str.startswith('[#x2014;]')
    bullet_mask = bullet_mask | corrected_df['text'].str.strip().str.startswith('[#x2022;]')
    corrected_df.loc[bullet_mask, 'FinalClass'] = TextClass.BULLET_LIST

    paragraph_classes = [TextClass.PARAGRAPH, TextClass.PARAGRAPH_CONT]
    # If previous line or next line is BULLET_LIST and current line is PARAGRAPH, and current line starts with a bullet, then current line is BULLET_LIST
    bullet_regex = r'^(\[\#\w+;\]|[#•]+\s)'
    bullet_classes = [TextClass.BULLET_LIST, TextClass.BULLET_LIST_CONT]
    corrected_df.loc[
        (corrected_df['FinalClass'].isin(paragraph_classes))
        & ((prev_line.isin(bullet_classes)) | (next_line.isin(bullet_classes)))
        & corrected_df['text'].str.match(bullet_regex),
        'FinalClass',
    ] = TextClass.BULLET_LIST

    # If previous line or next line is ENUM_LIST and current line is PARAGRAPH, and current line starts with a number, then current line is ENUM_LIST

    enum_classes = [TextClass.ENUM_LIST, TextClass.ENUM_LIST_CONT]
    corrected_df.loc[
        (corrected_df['FinalClass'].isin(paragraph_classes))
        & ((prev_line.isin(enum_classes)) | (next_line.isin(enum_classes)))
        & corrected_df['text'].str.match(enum_regex),
        'FinalClass',
    ] = TextClass.ENUM_LIST

    # If current line is BULLET_LIST and does not start with a bullet, then current line is PARAGRAPH
    corrected_df.loc[
        (corrected_df['FinalClass'] == TextClass.BULLET_LIST) & ~corrected_df['text'].str.match(bullet_regex),
        'FinalClass',
    ] = TextClass.PARAGRAPH

    # If current line is ENUM_LIST and does not start with a number, then current line is PARAGRAPH
    # corrected_df.loc[
    #     (corrected_df['FinalClass'] == TextClass.ENUM_LIST)
    #     & ~corrected_df['text'].str.match(enum_regex),
    #     'FinalClass',
    # ] = TextClass.PARAGRAPH

    # If previous line is BULLET_LIST and next line is BULLET_LIST, and current line is NOT a Paragraph first line or a heading first Line, current line = BULLET_LIST_CONT
    # corrected_df.loc[
    #     (prev_line == TextClass.BULLET_LIST)
    #     & (next_line == TextClass.BULLET_LIST)
    #     & (corrected_df['FinalClass'] != TextClass.PARAGRAPH)
    #     & (corrected_df['FinalClass'] != TextClass.HEADING),
    #     'FinalClass',
    # ] = TextClass.BULLET_LIST_CONT

    # If any "cont" class does not follow a "first_line" or "cont" class of the same type, it should be changed to be the continued version of the class of the previous line
    # (i.e. paragraph_cont does not have "paragraph" or "paragraph_cont" as the previous line)
    def get_continue_pair(textclass: TextClass) -> TextClass:
        for base_class, cont_class in CONTINUE_PAIRS:
            if textclass == base_class:
                return cont_class
        return None

    # Find lines that are "cont" classes but don't follow the right pattern
    for textclass, textclass_cont in CONTINUE_PAIRS:
        # Get the previous line's class for each row
        prev_line_class = corrected_df['FinalClass'].shift(1)

        # Find "cont" lines that don't follow a base class or continue class of the same type
        mask = (
            (corrected_df['FinalClass'] == textclass_cont)
            & (prev_line_class != textclass)  # Previous line is not the base class
            & (prev_line_class != textclass_cont)  # Previous line is not the continue class
        )

        # Apply the correction: set to the continue version of the previous line's class
        # For each row in the mask, get the continue version of its previous line's class
        for idx in corrected_df[mask].index:
            prev_class = prev_line.loc[idx]
            if pd.notna(prev_class):
                continue_version = get_continue_pair(prev_class)
                if continue_version is not None:
                    corrected_df.loc[idx, 'FinalClass'] = continue_version

    corrected_df['FinalClass'] = corrected_df['FinalClass'].astype(int)
    # add back rows where the 'FinalClass' is not in AI_PARSED_CLASSES
    corrected_df = pd.concat([corrected_df, df[~df['PredictedClass'].isin(AI_PARSED_CLASSES)]])
    corrected_df = corrected_df.sort_values(by='original_index')
    corrected_df = corrected_df.drop(columns=['original_index'])

    return corrected_df
