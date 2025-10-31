from enum import IntEnum


class TextClass(IntEnum):
    PARAGRAPH = 0
    PARAGRAPH_CONT = 1
    HEADING = 2
    HEADING_CONT = 3
    TOC = 4
    TABLE = 5
    HEADER = 6
    FOOTER = 7
    FOOT_NOTE = 8
    DEFINITION = 9
    BULLET_LIST = 10
    ENUM_LIST = 11
    GEN_LIST_CONT = 12
    BULLET_LIST_CONT = 13
    ENUM_LIST_CONT = 14


AI_PARSED_CLASSES = [
    TextClass.PARAGRAPH,
    TextClass.PARAGRAPH_CONT,
    TextClass.HEADING,
    TextClass.HEADING_CONT,
    TextClass.TOC,
    # TextClass.BULLET_LIST,
    # TextClass.ENUM_LIST,
    # TextClass.GEN_LIST_CONT,
]


CONTINUE_PAIRS = [
    (TextClass.HEADING, TextClass.HEADING_CONT),
    (TextClass.PARAGRAPH, TextClass.PARAGRAPH_CONT),
    (TextClass.ENUM_LIST, TextClass.ENUM_LIST_CONT),
    (TextClass.BULLET_LIST, TextClass.BULLET_LIST_CONT),
    (TextClass.GEN_LIST_CONT, TextClass.GEN_LIST_CONT),
]

CLASS_MAP = {value: idx for idx, value in enumerate(AI_PARSED_CLASSES)}
CLASS_MAP_INV = {idx: value for idx, value in enumerate(AI_PARSED_CLASSES)}
