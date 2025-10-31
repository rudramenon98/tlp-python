import logging
from pathlib import Path
from typing import Callable

import pandas as pd
from ai_doc_parser import DATA_DIR
from ai_doc_parser.text_class import TextClass

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def apply_heuristic_to_dir(
    dir_path: Path,
    heuristic_function: Callable[[pd.DataFrame], pd.DataFrame],
    output_dir: Path = None,
) -> None:
    """
    Apply a heuristic function to all the files in a directory
    """
    if output_dir is None:
        output_dir = dir_path
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list(dir_path.glob("*.csv"))
    for file in files:
        print(f"Applying heuristic to {file}")
        log.info("Applying heuristic to %s", file)
        df = pd.read_csv(file)
        df = heuristic_function(df)
        df.to_csv(file, index=False)


def latex_heuristics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply heuristics to the dataframe
    """
    # if the font_size > 18, then it is a heading
    df.loc[df["font_size"] > 12, "LabelledClass"] = TextClass.HEADING.value
    df.loc[df["font_size"] > 12, "LabelledClassName"] = TextClass.HEADING.name

    return df


def main() -> None:
    data_dir = DATA_DIR / "documents"
    latex_dir = data_dir / "Latex" / "labelled_pdf"
    latex_files = list(latex_dir.glob("*.csv"))

    output_dir = data_dir / "Latex" / "labelled_pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    for latex_file in latex_files:
        df = pd.read_csv(latex_file)

        df = latex_heuristics(df)

        output_file = output_dir / f"{latex_file.stem}.csv"
        df.to_csv(output_file, index=False)

        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
