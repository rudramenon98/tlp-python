import logging
import re
from collections import defaultdict

log = logging.getLogger(__name__)

# import requests
# import urllib.request
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import lxml.etree as ET
import numpy as np
import pandas as pd
from ai_doc_parser.text_class import TextClass
from ai_doc_parser.training.label_extractors.extractor_methods import check_for_columns

log = logging.getLogger(__name__)

# New XML Parser by Darpan

"""Roman Numerical Values Convert to Numbers"""
romanpages = {
    "i": 1,
    "?ii": 2,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
    "xi": 11,
    "xii": 12,
    "xiii": 13,
    "xiv": 14,
    "xv": 15,
    "xvi": 16,
    "xvii": 17,
    "xviii": 18,
    "xix": 19,
    "xx": 20,
    "xxi": 21,
    "xxii": 22,
    "xxiii": 23,
    "xxiv": 24,
    "xxv": 25,
    "xxvi": 26,
    "xxvii": 27,
    "xxviii": 28,
    "xxix": 29,
    "xxx": 30,
}

"""Setting Parameters"""
romanlist = set()
romancount = 0
integerlist = set()

totalRomanPages = 0
totalIntegerPages = 0

decimalBeforeRoman = False
minRomanSourceline = 10000000000
srcline2PageDict = dict()

missingRomanList = []
missingIntegerList = []

"""Calculation of roman meta Data"""


def calculate_roman_meta_data(df2: List[ET._Element]) -> None:
    global romanpages, romanlist, romancount, integerlist, totalRomanPages, totalIntegerPages
    count = 0

    for t in df2:
        try:
            pagenoText = t.attrib["P"]
        except KeyError:
            continue
        # print("Parsed page number text = " + pagenoText)
        count += 1
        try:
            j = int(pagenoText)
            integerlist.add(j)
            # print(str(count) + ' Page number is integer ' + str(j))
        except ValueError:
            if pagenoText not in romanpages.keys():
                continue
            # print(str(count) +  ' Page number is roman ' + pagenoText)
            romanlist.add(romanpages[pagenoText])
            offset = romanpages[pagenoText] - count
            # print('offset =' + str(offset))
            if offset > 0:
                count += offset
            romancount += 1

    totalRomanPages = len(romanlist) + len(find_missing(list(romanlist)))
    totalIntegerPages = len(integerlist) + len(find_missing(list(integerlist)))


def getPrevRomanNumber(rNo: int) -> int:
    idx = 0
    for key, val in romanpages.items():
        if val == rNo:
            return romanpages[list(romanpages.keys())[idx - 1]]
        idx += 1
    return 0  # Default return value if not found


def find_missing(lst: List[int]) -> List[int]:
    if len(lst) > 0:
        # return sorted(set(range(lst[0], lst[-1])) - set(lst))
        return sorted(set(range(lst[0], lst[-1])) - set(lst))
    else:
        return []


decimalIntsBeforeRoman = {}


def calculate_roman_meta_data2(df2: List[ET._Element]) -> None:
    global romanpages, romanlist, romancount, integerlist, totalRomanPages, totalIntegerPages, decimalBeforeRoman, minRomanSourceline, missingIntegerList, missingRomanList

    count = 0
    prevPageNo = -1
    prevPageSourceline = 0

    for idx, t in enumerate(df2):
        try:
            pagenoText = t.attrib["P"]
        except KeyError:
            continue
        # print("Parsed page number text = " + pagenoText)
        count += 1
        try:
            j = int(pagenoText)
            integerlist.add(j)
            prevPageNo = j
            prevPageSourceline = t.sourceline
            # print(str(count) + ' Page number is integer ' + str(j))
        except ValueError:
            if pagenoText not in romanpages.keys():
                continue
            # print(str(count) +  ' Page number is roman ' + pagenoText)
            romanlist.add(romanpages[pagenoText])
            offset = romanpages[pagenoText] - count
            # print('offset =' + str(offset))
            if offset > 0:
                count += offset
            romancount += 1
            if t.sourceline < minRomanSourceline:
                minRomanSourceline = t.sourceline
            if prevPageNo > 0 and not decimalBeforeRoman:
                decimalBeforeRoman = True
                decimalIntsBeforeRoman[prevPageSourceline] = prevPageNo

    if decimalBeforeRoman:
        totalRomanPages = (
            len(romanlist)
            + len(find_missing(list(romanlist)))
            + len(decimalIntsBeforeRoman)
        )
    else:
        totalRomanPages = len(romanlist) + len(find_missing(list(romanlist)))
    totalIntegerPages = len(integerlist) + len(find_missing(list(integerlist)))
    missingRomanList = find_missing(list(romanlist))
    missingIntegerList = find_missing(list(integerlist))


def populate_srcline2pageDict(df2: List[ET._Element], nLinesInFile: int) -> None:
    count = 0
    prevPageSourceline = 0
    prevPageNoText = ""

    for idx, t in enumerate(df2):
        try:
            count += 1
            pagenoText = t.attrib["P"]
            srcLine = t.sourceline
        except KeyError:
            continue
        # print("Parsed page number text = " + pagenoText)
        if idx + 1 == len(df2):
            for i in range(prevPageSourceline, srcLine):
                srcline2PageDict[i] = prevPageNoText
            for i in range(srcLine, nLinesInFile):
                srcline2PageDict[i] = pagenoText
        elif count > 1:
            for i in range(prevPageSourceline, srcLine):
                srcline2PageDict[i] = prevPageNoText
        else:
            for i in range(1, srcLine):
                srcline2PageDict[i] = pagenoText

        prevPageSourceline = srcLine
        prevPageNoText = pagenoText


def getCorrectPageNumber(textPageNo: str) -> int:
    # check if it is roman or integer
    global romanpages, romanlist, romancount, integerlist, totalRomanPages, totalIntegerPages
    try:
        j = int(textPageNo)
        # handle integer values
        ioffset = 1 - min(integerlist)
        for x in find_missing(list(integerlist)):
            if x < j:
                ioffset += 1
        return j + ioffset + totalRomanPages
    except ValueError:
        # handle roman values
        intPageNo = romanpages[textPageNo]  # need to check for index error!
        roffset = 0
        for x in find_missing(list(romanlist)):
            if x < intPageNo:
                roffset += 1
        return intPageNo + roffset


def getCorrectPageNumber2(textPageNo: str, srcline: int) -> int:
    # check if it is roman or integer
    global romanpages, romanlist, romancount, integerlist, totalRomanPages, totalIntegerPages, decimalBeforeRoman, minRomanSourceline, missingIntegerList, missingRomanList

    try:
        j = int(textPageNo)
        if decimalBeforeRoman and srcline < minRomanSourceline:
            val = getPrevRomanNumber(list(romanlist)[0])
            roffset = 0
            for x in missingRomanList:
                if x < val:
                    roffset += 1
            return val + roffset

        # handle integer values
        if decimalBeforeRoman:
            ioffset = 0
        else:
            ioffset = 1 - min(integerlist)
            for x in missingIntegerList:
                if x < j:
                    ioffset += 1
        return j + ioffset + totalRomanPages
    except ValueError:
        # handle roman values
        intPageNo = romanpages[textPageNo]  # need to check for index error!
        if decimalBeforeRoman:
            roffset = 0
            return intPageNo
        else:
            roffset = 0
            for x in missingRomanList:
                if x < intPageNo:
                    roffset += 1
            return intPageNo + roffset


def getCorrectPageNumber3(inputtextPageNo: str, srcline: int) -> int:
    # check if it is roman or integer
    global romanpages, romanlist, romancount, integerlist, totalRomanPages, totalIntegerPages, decimalBeforeRoman, minRomanSourceline, missingIntegerList, missingRomanList

    if srcline == 256:
        log.debug("Debug point reached")
    textPageNo = srcline2PageDict.get(srcline, None)
    if textPageNo is None:
        log.debug("ERROR: Page Number Not Found for source line:" + str(srcline))
        return -1
    #    elif textPageNo != inputtextPageNo:
    #        print('ERROR: Page Number data wrong for source line:' + str(srcline) )
    #        print('Found :' + textPageNo + 'instead of ' + inputtextPageNo)
    #        return -1

    try:
        j = int(textPageNo)
        if decimalBeforeRoman and srcline < minRomanSourceline:
            val = getPrevRomanNumber(list(romanlist)[0])
            roffset = 0
            for x in missingRomanList:
                if x < val:
                    roffset += 1
            return val + roffset

        # handle integer values
        if decimalBeforeRoman:
            ioffset = 0
        else:
            ioffset = 1 - min(integerlist)
            for x in missingIntegerList:
                if x < j:
                    ioffset += 1
        return j + ioffset + totalRomanPages
    except ValueError:
        # handle roman values
        intPageNo = romanpages[textPageNo]  # need to check for index error!
        if decimalBeforeRoman:
            roffset = 0
            return intPageNo
        else:
            roffset = 0
            for x in missingRomanList:
                if x < intPageNo:
                    roffset += 1
            return intPageNo + roffset


def getCorrectPageNumberN(inputtextPageNo: str, srcline: int) -> int:
    # check if it is roman or integer
    global romanpages, romanlist, romancount, integerlist, totalRomanPages, totalIntegerPages

    textPageNo = srcline2PageDict.get(srcline, None)
    if textPageNo is None:
        log.debug("ERROR: Page Number Not Found for source line:" + str(srcline))
        return -1
    #    elif textPageNo != inputtextPageNo:
    #        print('ERROR: Page Number data wrong for source line:' + str(srcline) )
    #        print('Found :' + textPageNo + 'instead of ' + inputtextPageNo)
    #        return -1

    try:
        j = int(textPageNo)
        if decimalBeforeRoman and srcline < minRomanSourceline:
            return j
        # handle integer values
        ioffset = 1 - min(integerlist)
        #        for x in find_missing(list(integerlist)):
        #            if x < j:
        #                ioffset += 1
        return j + ioffset + totalRomanPages
    except ValueError:
        # handle roman values
        intPageNo = romanpages[textPageNo]  # need to check for index error!
        roffset = 0
        #        for x in find_missing(list(romanlist)):
        #            if x < intPageNo:
        #                roffset += 1
        return intPageNo + roffset


def checkAncestors(curr_tag: ET._Element, search_tag: str) -> bool:
    for p in curr_tag.iterancestors():
        if p.tag == search_tag:
            return True
    return False


def getSiblings(
    curr_tag: ET._Element, tags: Optional[List[str]] = None
) -> List[ET._Element]:
    siblings = []
    for s in curr_tag.iterancestors():
        if tags:
            if s.tag in tags:
                siblings.append(s)
        else:
            siblings.append(s)
    return siblings


def getChildren(
    curr_tag: ET._Element, tags: Optional[List[str]] = None
) -> List[ET._Element]:
    children = []
    for s in curr_tag.iterchildren():
        if tags:
            if s.tag in tags:
                children.append(s)
        else:
            children.append(s)

    return children


def getChildren_D(
    curr_tag: ET._Element, tags: Optional[List[str]] = None
) -> List[ET._Element]:
    children = []
    # for s in curr_tag.iterchildren():
    for s in curr_tag.getiterator():
        if tags:
            children.append(s)

    return children


def linecount(filename: Path) -> int:
    lines = 0
    try:
        # Try UTF-8 first
        for line in open(filename, encoding="utf-8"):
            lines += 1
    except UnicodeDecodeError:
        try:
            # Try UTF-8 with BOM
            for line in open(filename, encoding="utf-8-sig"):
                lines += 1
        except UnicodeDecodeError:
            try:
                # Try latin-1 (fallback for Windows)
                for line in open(filename, encoding="latin-1"):
                    lines += 1
            except UnicodeDecodeError:
                # Last resort: try with errors='ignore'
                for line in open(filename, encoding="utf-8", errors="ignore"):
                    lines += 1
    return lines


@check_for_columns
def cfr_extracting(path: Path) -> pd.DataFrame:
    """
    ***Important***
    Path : Path of the file

    * This function will take path as an argument and return the parsed xml file

    """
    # check if the file exists
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist")

    start = time.time()

    xml_file_name = path
    tree = ET.parse(xml_file_name)
    root = tree.getroot()

    # Count number of lines in file
    num_lines_in_file = linecount(xml_file_name)

    # Get page number
    df2 = root.findall(".//PRTPAGE")
    calculate_roman_meta_data2(df2)
    populate_srcline2pageDict(df2, num_lines_in_file)

    cfr_tags_path = Path(__file__).parent / "CFR_Tags.xlsx"
    df = pd.read_excel(cfr_tags_path, header=None)
    df2 = df.rename(columns=df.iloc[2]).loc[3:]

    # Initialize dictionary to store all data
    data_dict: Dict[str, List[Any]] = defaultdict(list)

    # regex for beginning of list items
    list_item_patterns = [
        r"^\(\s*[a-zA-Z0-9]+\s*\) ",  # (a), (1)
        r"^[a-zA-Z]\) ",  # a)
        r"^[0-9]+\) ",  # 1)
        r"^[ivxlcdmIVXLCDM]+\) ",  # i)
        r"^[0-9]+\.[\s)]? ",  # 1. or 1. )
        r"^[a-zA-Z]\. ",  # a.
        r"^[ivxlcdmIVXLCDM]+\. ",  # i.
    ]

    def is_first_item(match: Optional[re.Match]) -> bool:
        if match:
            marker = match.group(0).lower()
            # remove all the special characters
            marker = re.sub(r"[^a-zA-Z0-9]", "", marker)
            marker = marker.strip()
            return marker in ("1", "a", "i")
        return False

    page_no_text = "i"
    page_no = 0

    heading_tags = ["PART", "HD", "TITLENUM", "HEADING", "SECTNO", "SUBJECT"]
    tags_to_skip = list(df2["Non Text tags"])

    # Filter out specific tags that should not be skipped
    tags_to_skip = [i for i in tags_to_skip if i not in ["TOCTAC", "FTNT", "EXTRACT"]]

    def append_to_dict(
        elem: ET._Element,
        text: str,
        heading_val: int,
        paragraph_val: int,
        toc_val: int,
        table_val: int,
        preamble_val: int,
        class_val: int,
        footnote_val: int = 0,
    ) -> None:
        """Helper function to append data to the dictionary"""
        data_dict["LineNumbers"].append(elem.sourceline)
        data_dict["tag_list"].append(elem.tag)
        data_dict["text"].append(text)
        data_dict["heading"].append(heading_val)
        data_dict["paragraph"].append(paragraph_val)
        data_dict["TOC"].append(toc_val)
        data_dict["Table"].append(table_val)
        # data_dict['Table'].append(0)
        data_dict["Pages"].append(page_no_text)
        data_dict["PageNumber"].append(
            getCorrectPageNumberN(page_no_text, elem.sourceline)
        )
        data_dict["Preamble"].append(preamble_val)
        # if class_val in [TextClass.TABLE, TextClass.FOOTER, TextClass.HEADER]:
        # class_val = TextClass.PARAGRAPH
        data_dict["SourceClass"].append(class_val)
        data_dict["Footnote"].append(footnote_val)
        # data_dict['Footnote'].append(0)

    for elem in tree.iter():
        # Skip if already processed as part of a higher tag
        if elem.sourceline in data_dict["LineNumbers"]:
            continue

        # Handle tags that should be skipped
        if elem.tag in tags_to_skip:
            page_tags = elem.find(".//PRTPAGE")
            if page_tags is not None and page_tags.tag == "PRTPAGE":
                page_no_text = page_tags.attrib["P"]
                page_no += 1

        toc = 0
        table = 0
        preamble = 0

        # Handle page number tags
        if elem.tag == "PRTPAGE":
            if elem.attrib:
                page_no_text = elem.attrib["P"]
                page_no += 1
            continue

        # Handle heading tags
        if elem.tag in heading_tags:
            append_to_dict(
                elem, elem.text.strip(), 1, 0, toc, table, preamble, TextClass.HEADING
            )
            continue

        # Handle table of contents tags
        if elem.tag in ["TOC", "CFRTOC"]:
            toc = 1
            if elem.tag == 31146:
                log.debug("Debug point reached")
            for k in elem:
                if k.tag == "PRTPAGE" and k.sourceline not in data_dict["LineNumbers"]:
                    page_no_text = k.attrib["P"]
                    page_no += 1
                for j in k:
                    if (
                        j.tag == "PRTPAGE"
                        and j.sourceline not in data_dict["LineNumbers"]
                    ):
                        page_no_text = j.attrib["P"]
                        continue
                    text = "".join(j.itertext())
                    if j.tag in heading_tags:
                        append_to_dict(
                            j, text, 1, 0, 0, table, preamble, TextClass.HEADING
                        )
                    elif j.tag == "P":
                        append_to_dict(
                            j, text, 0, 1, 0, table, preamble, TextClass.PARAGRAPH
                        )
                    else:
                        append_to_dict(j, text, 0, 0, 1, table, preamble, TextClass.TOC)
                    continue
            continue

        # Handle table tags
        if elem.tag in ["GPOTABLE", "ROW"]:
            table = 1
            children = getChildren_D(elem, [str(elem.tag)])
            for table_elem in children:
                text = "".join(table_elem.itertext())
                append_to_dict(
                    table_elem, text, 0, 0, toc, table, preamble, TextClass.TABLE
                )
            continue

        # Handle paragraph tags
        if elem.tag == "P":
            is_list_item = False
            text = "".join(elem.itertext())
            match = None
            for pattern in list_item_patterns:
                match = re.match(pattern, text)
                if match:
                    is_list_item = True
                    break
            if is_list_item:
                append_to_dict(
                    elem, elem.text, 0, 1, toc, table, preamble, TextClass.PARAGRAPH
                )
            else:
                append_to_dict(
                    elem, text, 0, 1, toc, table, preamble, TextClass.PARAGRAPH
                )
            continue

        # Skip E tags
        if elem.tag == "E":
            continue

        # Handle AUTH tags
        if elem.tag == "AUTH":
            preamble = 1
            children = getChildren_D(elem, ["AUTH"])

            for child in children:
                if (
                    child.tag not in tags_to_skip
                    and child.text
                    and str(child.text) != "nan"
                    and len(child.text.strip()) > 0
                ):
                    text = child.text.strip()
                    append_to_dict(
                        child, text, 0, 1, toc, table, preamble, TextClass.PARAGRAPH
                    )
            continue

        # Handle TOCTAC tags
        if elem.tag == "TOCTAC":
            log.debug("here")
            preamble = 1
            children = getChildren_D(elem, ["TOCTAC"])

            for child in children:
                if (
                    child.tag not in tags_to_skip
                    and child.text
                    and str(child.text) != "nan"
                    and len(child.text.strip()) > 0
                ):
                    text = child.text.strip()
                    append_to_dict(
                        child, text, 0, 1, toc, table, preamble, TextClass.PARAGRAPH
                    )
            continue

        # Handle CONTENTS tags
        if elem.tag == "CONTENTS":
            preamble = 1
            children = getChildren_D(elem, ["CONTENTS"])

            for child in children:
                if (
                    child.tag not in tags_to_skip
                    and child.text
                    and str(child.text) != "nan"
                    and len(child.text.strip()) > 0
                ):
                    text = child.text.strip()
                    append_to_dict(
                        child, text, 0, 1, toc, table, preamble, TextClass.PARAGRAPH
                    )
            continue

        # Handle EXTRACT tags
        if elem.tag == "EXTRACT":
            preamble = 0
            children = getChildren_D(elem, ["EXTRACT"])

            for child in children:
                if (
                    child.tag not in tags_to_skip
                    and child.text
                    and str(child.text) != "nan"
                    and len(child.text.strip()) > 0
                ):
                    text = "".join(child.itertext())
                    append_to_dict(
                        child, text, 1, 0, toc, table, preamble, TextClass.PARAGRAPH
                    )
            continue

        # Handle footnote tags
        if elem.tag == "FTNT":
            text = "".join(elem.itertext())
            append_to_dict(elem, text, 0, 1, toc, table, preamble, TextClass.FOOTER)
            continue

        # Handle remaining text elements
        if elem.text and str(elem.text) != "nan" and len(elem.text.strip()) > 0:
            if elem.tag in heading_tags:
                heading_val = 1
                paragraph_val = 0
                class_val = 2
            elif elem.tag in heading_tags:
                heading_val = 0
                paragraph_val = 1
                class_val = 0
            else:
                parent_tag = elem.getparent()
                if parent_tag is not None and parent_tag.tag in heading_tags:
                    heading_val = 1
                    paragraph_val = 0
                    class_val = 2
                else:
                    heading_val = 0
                    paragraph_val = 1
                    class_val = 0

            page_number = (
                getCorrectPageNumberN(page_no_text, elem.sourceline)
                if len(page_no_text.strip()) > 0
                else 1
            )

            append_to_dict(
                elem,
                elem.text,
                heading_val,
                paragraph_val,
                toc,
                table,
                preamble,
                class_val,
            )
            # Update the page number for this specific entry
            data_dict["PageNumber"][-1] = page_number

    # Create DataFrame from collected data
    raw_form = pd.DataFrame(data_dict)

    # Add temporary columns for text processing
    raw_form["child_tag_temp"] = raw_form["tag_list"].shift(-1)
    raw_form["child_tag_temp_2"] = raw_form["tag_list"].shift(-2)

    raw_form["child_tag_text"] = raw_form["text"].shift(-1)
    raw_form["child_tag_text_2"] = raw_form["text"].shift(-2)

    raw_form["child_tag_line_num"] = raw_form["LineNumbers"].shift(-1)
    raw_form["child_tag_line_num_2"] = raw_form["LineNumbers"].shift(-2)

    # Process TOC entries and combine related text
    remove_vals = []
    for idx, row in raw_form.iterrows():
        # Combine PT, SUBJECT, and PG tags in TOC
        if (
            row["TOC"] == 1
            and row["tag_list"] == "PT"
            and row["child_tag_temp"] == "SUBJECT"
            and row["child_tag_temp_2"] == "PG"
        ):
            raw_form.at[idx, "text"] = (
                row["text"]
                + " "
                + " "
                + row["child_tag_text"]
                + " "
                + row["child_tag_text_2"]
            )
            remove_vals.append((row["child_tag_line_num"]))
            remove_vals.append(row["child_tag_line_num_2"])

        # Combine SUBJECT and PG tags in TOC
        if (
            row["TOC"] == 1
            and row["tag_list"] == "SUBJECT"
            and row["child_tag_temp"] == "PG"
            and row["child_tag_temp_2"] != "PT"
        ):
            raw_form.at[idx, "text"] = row["text"] + " " + " " + row["child_tag_text"]
            raw_form.at[idx, "heading"] = 0
            raw_form.at[idx, "SourceClass"] = 2
            remove_vals.append((row["child_tag_line_num"]))

    # Process section number entries
    remove_vals_sec = []
    for idx, row in raw_form.iterrows():
        # Combine SECTNO and SUBJECT tags
        if row["tag_list"] == "SECTNO" and row["child_tag_temp"] == "SUBJECT":
            raw_form.at[idx, "text"] = row["text"] + " " + " " + row["child_tag_text"]
            remove_vals_sec.append((row["child_tag_line_num"]))

    # Remove processed entries
    raw_form = raw_form[~raw_form["LineNumbers"].isin(remove_vals + remove_vals_sec)]

    # Clean and normalize text
    raw_form.text = raw_form.text.apply(lambda x: str(x).strip())
    raw_form.text = raw_form.text.apply(lambda x: " ".join(x.split()))
    raw_form = raw_form.apply(lambda x: x.strip() if isinstance(x, str) else x).replace(
        "", np.nan
    )

    # Sort by line numbers and remove temporary columns
    raw_form = raw_form.sort_values(["LineNumbers"], ascending=True)
    raw_form["xml_idx"] = raw_form.index

    # Keep only important columns
    raw_form = raw_form[["LineNumbers", "text", "PageNumber", "SourceClass", "xml_idx"]]
    raw_form["SourceClassName"] = raw_form["SourceClass"].apply(
        lambda x: TextClass(x).name
    )

    # Remove rows with missing data and save to CSV
    raw_form = raw_form.dropna()

    end = time.time()
    log.debug("Time for XML parsing = " + str(end - start))
    log.debug("SUCCESS")
    return raw_form


def split_paragraphs_into_chunks(text: str, n_words: int) -> List[str]:
    # split the text
    pieces = text.split()

    # return the chunks
    return list(
        " ".join(pieces[i : i + n_words]) for i in range(0, len(pieces), n_words)
    )


def main() -> None:
    data_dir = Path(__file__).parents[4] / "data" / "documents" / "CFR"
    output_dir = data_dir / "labelled_source"
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_path = data_dir / "CFR-2025-title14-vol1.xml"
    log.debug("XML path exists: %s, path: %s", xml_path.exists(), xml_path)

    df = cfr_extracting(xml_path)
    df.to_csv(output_dir / f"{xml_path.stem}.csv", index=False)
    log.info("Saved to %s", output_dir / f"{xml_path.stem}.csv")


if __name__ == "__main__":
    main()
