from typing import Any, Callable

import pandas as pd


def check_for_columns(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> pd.DataFrame:
        columns = ['LineNumbers', 'text', 'SourceClass', 'SourceClassName', 'PageNumber', 'xml_idx']
        df = func(*args, **kwargs)
        if not all(col in df.columns for col in columns):
            raise ValueError(f"Columns {columns} not found in {df.columns}")
        return df

    return wrapper
