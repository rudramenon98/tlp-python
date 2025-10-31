import re
from functools import lru_cache


@lru_cache(maxsize=10000)
def clean_text(text: str) -> str:
    """
    Clean and normalize a string by removing special characters and converting to lowercase.
    This method combines all cleaning operations into a single comprehensive function.
    Cached for performance - same strings don't need to be cleaned multiple times.

    Args:
        x: Input string to clean

    Returns:
        Cleaned and normalized string
    """

    text = re.sub(r"\[#\w+;\]", "", text)
    # only keep alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()

    # remove leading and trailing spaces
    text = text.strip()
    # while double spaces exist, replace them with single spaces
    while "  " in text:
        text = text.replace("  ", " ")

    return text
