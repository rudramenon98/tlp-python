import logging

import pandas as pd


from ai_doc_parser.text_class import TextClass
from ai_doc_parser.inference.feature_computation.feature_computer import *
feature_functions = {
        "left_indent": left_indent,
        "right_space": right_space,
        "is_bold": is_bold,
        "is_italic": is_italic,
        "font_size": font_size,
        "font_color": font_color,
        "first_char_isdigit": first_char_isdigit,
        "last_char_isdigit": last_char_isdigit,
        "first_character_is_bullet": first_character_is_bullet,
        "first_character_is_upper": first_character_is_upper,
        "last_character_is_upper": last_character_is_upper,
        "first_word_is_compound": first_word_is_compound,
        "starts_with_keyword": starts_with_keyword,
        "number_dots": number_dots,
        "no_of_words": no_of_words,
        "fraction_capitalized": fraction_capitalized,
        "ends_with_period": ends_with_period,
        "ends_with_question_mark": ends_with_question_mark,
        "ends_with_exclamation": ends_with_exclamation,
        "ends_with_dash_characters": ends_with_dash_character,
        "ends_with_punctuation": ends_with_punctuation,
        "ends_with_letter": ends_with_letter,
        "ends_with_other_special_characters": ends_with_other_special_characters,
        "ends_with_digit": ends_with_digit,
        "last_word_is_roman": last_word_is_roman,
        "ends_with_upper": ends_with_upper,
        "no_count_consecutive_spaces": no_count_consecutive_spaces,
        "consecutive_dots": consecutive_dots,
        "replace_space": replace_space,
    }


if __name__ == "__main__":
    from ai_doc_parser import DATA_DIR
    from ai_doc_parser import EASA_PDF as pdf_path
    
    

    df_path = pdf_path.parent / "labelled_pdf" / f"{pdf_path.stem}.csv"

    logging.basicConfig(level=logging.DEBUG)

    df = pd.read_csv(df_path)
    print(df.columns)
    df = df[df["ClassLabel"] == TextClass.BULLET_LIST]
    for i, row in df.iterrows():
        print('-----')
        print(row["text"])
        for key, func in feature_functions.items():
            print(f"\t{key}: {func(row)}")
