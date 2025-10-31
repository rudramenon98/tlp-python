import logging
import re
import time
from pathlib import Path

log = logging.getLogger(__name__)

import lxml.etree as ET
import numpy as np
import pandas as pd
from ai_doc_parser.text_class import TextClass
from ai_doc_parser.training.label_extractors.extractor_methods import check_for_columns

log = logging.getLogger(__name__)

# New XML Parser by Darpan

'''Roman Numerical Values Convert to Numbers'''
romanpages = {
    'i': 1,
    '?ii': 2,
    'ii': 2,
    'iii': 3,
    'iv': 4,
    'v': 5,
    'vi': 6,
    'vii': 7,
    'viii': 8,
    'ix': 9,
    'x': 10,
    'xi': 11,
    'xii': 12,
    'xiii': 13,
    'xiv': 14,
    'xv': 15,
    'xvi': 16,
    'xvii': 17,
    'xviii': 18,
    'xix': 19,
    'xx': 20,
    'xxi': 21,
    'xxii': 22,
    'xxiii': 23,
    'xxiv': 24,
    'xxv': 25,
    'xxvi': 26,
    'xxvii': 27,
    'xxviii': 28,
    'xxix': 29,
    'xxx': 30,
}

'''Setting Parameters'''
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

'''Calculation of roman meta Data'''


def calculate_roman_meta_data(df2):
    global romanpages, romanlist, romancount, integerlist, totalRomanPages, totalIntegerPages
    count = 0

    for t in df2:
        try:
            pagenoText = t.attrib['P']
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


def getPrevRomanNumber(rNo):
    idx = 0
    for key, val in romanpages.items():
        if val == rNo:
            return romanpages[list(romanpages.keys())[idx - 1]]
        idx += 1


def find_missing(lst):
    if len(lst) > 0:
        # return sorted(set(range(lst[0], lst[-1])) - set(lst))
        return sorted(set(range(lst[0], lst[-1])) - set(lst))
    else:
        return []


decimalIntsBeforeRoman = {}


def calculate_roman_meta_data2(df2):
    global romanpages, romanlist, romancount, integerlist, totalRomanPages, totalIntegerPages, decimalBeforeRoman, minRomanSourceline, missingIntegerList, missingRomanList

    count = 0
    prevPageNo = -1
    prevPageSourceline = 0

    for idx, t in enumerate(df2):
        try:
            pagenoText = t.attrib['P']
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
        totalRomanPages = len(romanlist) + len(find_missing(list(romanlist))) + len(decimalIntsBeforeRoman)
    else:
        totalRomanPages = len(romanlist) + len(find_missing(list(romanlist)))
    totalIntegerPages = len(integerlist) + len(find_missing(list(integerlist)))
    missingRomanList = find_missing(list(romanlist))
    missingIntegerList = find_missing(list(integerlist))


def populate_srcline2pageDict(df2, nLinesInFile):
    count = 0
    prevPageSourceline = 0
    prevPageNoText = ''

    for idx, t in enumerate(df2):
        try:
            count += 1
            pagenoText = t.attrib['P']
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


def getCorrectPageNumber(textPageNo):
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


def getCorrectPageNumber2(textPageNo, srcline):
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


def getCorrectPageNumber3(inputtextPageNo, srcline):
    # check if it is roman or integer
    global romanpages, romanlist, romancount, integerlist, totalRomanPages, totalIntegerPages, decimalBeforeRoman, minRomanSourceline, missingIntegerList, missingRomanList

    if srcline == 256:
        log.debug()
    textPageNo = srcline2PageDict.get(srcline, None)
    if textPageNo is None:
        log.debug('ERROR: Page Number Not Found for source line:' + str(srcline))
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
        log.debug('ERROR: Page Number Not Found for source line:' + str(srcline))
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


def checkAncestors(curr_tag, search_tag):
    for p in curr_tag.iterancestors():
        if p.tag == search_tag:
            return True
    return False


def getSiblings(curr_tag, tags):
    siblings = []
    for s in curr_tag.iterancestors():
        if tags:
            if s.tag in tags:
                siblings.append(s)
        else:
            siblings.append(s)
    return siblings


def getChildren(curr_tag, tags=None):
    children = []
    for s in curr_tag.iterchildren():

        if tags:

            if s.tag in tags:
                children.append(s)
        else:
            children.append(s)

    return children


def getChildren_D(curr_tag, tags=None):
    children = []
    # for s in curr_tag.iterchildren():
    for s in curr_tag.getiterator():

        if tags:
            children.append(s)

    return children


def linecount(filename):
    lines = 0
    try:
        # Try UTF-8 first
        for line in open(filename, encoding='utf-8'):
            lines += 1
    except UnicodeDecodeError:
        try:
            # Try UTF-8 with BOM
            for line in open(filename, encoding='utf-8-sig'):
                lines += 1
        except UnicodeDecodeError:
            try:
                # Try latin-1 (fallback for Windows)
                for line in open(filename, encoding='latin-1'):
                    lines += 1
            except UnicodeDecodeError:
                # Last resort: try with errors='ignore'
                for line in open(filename, encoding='utf-8', errors='ignore'):
                    lines += 1
    return lines


def paragraph_text(elem):
    """Extract visible text from a <w:p> element."""
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    parts = []
    for child in elem.iter():
        if child.tag == f"{{{ns['w']}}}t":  # text node
            parts.append(child.text or "")
        elif child.tag == f"{{{ns['w']}}}tab":
            parts.append("\t")
        elif child.tag == f"{{{ns['w']}}}br":
            parts.append("\n")
    return "".join(parts)


@check_for_columns
def extract_easa_xml(path: Path) -> pd.DataFrame:
    '''
    ***Important***
    Path : Path of the file

    * This function will take path as an argument and return the parsed xml file

    '''
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
    df2 = root.findall('.//PRTPAGE')
    calculate_roman_meta_data2(df2)
    populate_srcline2pageDict(df2, num_lines_in_file)

    cfr_tags_path = Path(__file__).parent / "CFR_Tags.xlsx"
    df = pd.read_excel(cfr_tags_path, header=None)
    df2 = df.rename(columns=df.iloc[2]).loc[3:]

    # Initialize dictionary to store all data
    data_dict = {
        'LineNumbers': [],
        'OriginalPdfIndex': [],
        'tag_list': [],
        'text': [],
        'Pages': [],
        'PageNumber': [],
        'SourceClass': [],
        'SourceClassName': [],
        'style': [],
        'tag': [],
    }

    # regex for beginning of list items
    list_item_patterns = [
        r'^\(\s*[a-zA-Z0-9]+\s*\) ',  # (a), (1)
        r'^[a-zA-Z]\) ',  # a)
        r'^[0-9]+\) ',  # 1)
        r'^[ivxlcdmIVXLCDM]+\) ',  # i)
        r'^[0-9]+\.[\s)]? ',  # 1. or 1. )
        r'^[a-zA-Z]\. ',  # a.
        r'^[ivxlcdmIVXLCDM]+\. ',  # i.
    ]

    def is_first_item(match: re.Match) -> bool:
        if match:
            marker = match.group(0).lower()
            # remove all the special characters
            marker = re.sub(r'[^a-zA-Z0-9]', '', marker)
            marker = marker.strip()
            return marker in ("1", "a", "i")
        return False

    def get_list_level(elem) -> int:
        """
        Get the list level from a Word Open XML element.
        Returns the level number (0-based) or -1 if not a list item.
        """
        if elem.tag == 'w:p':
            # Use namespace-aware search or direct child iteration
            for child in elem.iterchildren():
                if str(child.tag).endswith('pPr'):
                    for grandchild in child.iterchildren():
                        if str(grandchild.tag).endswith('pStyle') and 'w:val' in grandchild.attrib:
                            style_val = grandchild.attrib['w:val']
                            if style_val and style_val.startswith('ListLevel'):
                                try:
                                    # Extract level number from "ListLevel0", "ListLevel1", etc.
                                    level = int(style_val.replace('ListLevel', ''))
                                    return level
                                except ValueError:
                                    return 0
        return -1

    def get_style_value(elem) -> str | None:
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        style = elem.find(".//w:pPr/w:pStyle", namespaces=ns)
        if style is not None:
            val = style.get(f"{{{ns['w']}}}val")  # w:val attribute
            return val
        return None

    def is_list_item(elem) -> bool:
        """
        Check if an XML element is part of a list by examining:
        1. Word Open XML paragraph styles (w:pStyle with ListLevel values)
        2. Text content patterns that indicate list items
        3. Presence of tab elements after list markers
        """
        # For debugging: you can get the raw XML text like this:
        # raw_xml = get_raw_xml_text(elem)
        # print(f"Raw XML: {raw_xml}")

        # Check for Word Open XML list styles
        # if elem.tag == 'w:p':
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        style = elem.find(".//w:pPr/w:pStyle", namespaces=ns)
        if elem.text and elem.text.strip():
            if style is not None:
                val = style.get(f"{{{ns['w']}}}val")  # w:val attribute
                if val and val.startswith("ListLevel"):
                    return True
        return False

    page_no_text = 'i'
    page_no = 0

    counter = 0

    def append_to_dict(
        elem: ET.Element,
        class_val: TextClass,
    ) -> None:
        """Helper function to append data to the dictionary"""
        nonlocal counter
        style = get_style_value(elem)
        tag = elem.tag.split("}")[-1]
        data_dict['LineNumbers'].append(elem.sourceline)
        data_dict['OriginalPdfIndex'].append(counter)
        data_dict['tag_list'].append(elem.tag)
        data_dict['text'].append(paragraph_text(elem))
        data_dict['Pages'].append(page_no_text)
        data_dict['PageNumber'].append(getCorrectPageNumberN(page_no_text, elem.sourceline))
        data_dict['SourceClass'].append(class_val.value)
        data_dict['SourceClassName'].append(class_val.name)
        data_dict['style'].append(style)
        data_dict['tag'].append(tag)
        counter += 1

    def str_in_list(s: str, l: list[str]) -> bool:
        if s is None:
            return False
        s = s.lower()
        l = [i.lower() for i in l if i is not None]
        return any(item in s for item in l)

    def is_placeholder(elem) -> bool:
        """
        Check if an XML element contains placeholder text by examining:
        1. Run style (w:rStyle) with PlaceholderText value
        2. Text content that matches placeholder patterns (e.g., [text])
        3. Presence in docPart structure
        """
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

        # Check for PlaceholderText style in run properties
        for run in elem.findall(".//w:r", namespaces=ns):
            r_style = run.find(".//w:rPr/w:rStyle", namespaces=ns)
            if r_style is not None:
                style_val = r_style.get(f"{{{ns['w']}}}val")
                if style_val and "PlaceholderText" in style_val:
                    return True

        # Check for placeholder text pattern (text in square brackets)
        text = paragraph_text(elem)
        if text and re.match(r'^\s*\[.*\]\s*$', text.strip()):
            return True

        # Check if element is inside a docPart structure
        if elem.xpath('boolean(ancestor::w:docPart)', namespaces=ns):
            return True

        return False

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    # unique_styles = set()
    # unique_tags = set()

    header_tags = []
    footnote_tags = ['footnote']
    heading_tags = ['TITLE', 'PART', 'HD', 'HEADING', 'SECTNO', 'SUBJECT']
    enum_list_tags = ['LISTLEVEL']
    bullet_list_tags = ['BULLET']
    footer_tags = ['FOOTER']
    placeholder_tags = ['PlaceholderText']
    tags_to_skip = list(df2['Non Text tags'])
    tags_to_skip = [i for i in tags_to_skip if i not in ['TOCTAC', 'FTNT', 'EXTRACT', 'E']]

    for elem in tree.iter("{%s}p" % ns["w"]):
        text = paragraph_text(elem)
        if not text.strip():
            continue

        # Skip if already processed as part of a higher tag
        # if elem.sourceline in data_dict['LineNumbers']:
        #     continue
        style = get_style_value(elem)
        tag = elem.tag.split("}")[-1]

        # print("--------------------------------")
        # print(text)
        # print(style)
        # print(tag)
        # Check for placeholders

        if str_in_list(style, heading_tags) or str_in_list(tag, heading_tags):
            append_to_dict(elem, TextClass.HEADING)
            continue

        if str_in_list(style, header_tags) or str_in_list(tag, header_tags):
            append_to_dict(elem, TextClass.HEADER)
            continue

        if str_in_list(style, footnote_tags) or str_in_list(tag, footnote_tags):
            append_to_dict(elem, TextClass.FOOT_NOTE)
            continue

        in_table = bool(elem.xpath('boolean(ancestor::w:tc)', namespaces=ns))
        if in_table:
            append_to_dict(elem, TextClass.TABLE)
            continue

        if is_placeholder(elem):
            append_to_dict(elem, TextClass.PARAGRAPH)
            continue

        if style and str_in_list(style, bullet_list_tags):
            append_to_dict(elem, TextClass.BULLET_LIST)
            continue

        if style and str_in_list(style, enum_list_tags):
            append_to_dict(elem, TextClass.ENUM_LIST)
            continue

        if style and "toc" in style.lower():
            append_to_dict(elem, TextClass.TOC)
            continue

        if str_in_list(style, footer_tags) or str_in_list(tag, footer_tags):
            append_to_dict(elem, TextClass.FOOTER)
            continue

        if str_in_list(style, tags_to_skip) or str_in_list(tag, tags_to_skip):
            continue
        # if style and "table" in style.lower():
        #     append_to_dict(elem, TextClass.TABLE)
        #     continue

        append_to_dict(elem, TextClass.PARAGRAPH)
        continue

    # Create DataFrame from collected data
    raw_form = pd.DataFrame(data_dict)

    # Add temporary columns for text processing
    raw_form['child_tag_temp'] = raw_form['tag_list'].shift(-1)
    raw_form['child_tag_temp_2'] = raw_form['tag_list'].shift(-2)

    raw_form['child_tag_text'] = raw_form['text'].shift(-1)
    raw_form['child_tag_text_2'] = raw_form['text'].shift(-2)

    raw_form['child_tag_line_num'] = raw_form['LineNumbers'].shift(-1)
    raw_form['child_tag_line_num_2'] = raw_form['LineNumbers'].shift(-2)

    # Process TOC entries and combine related text
    remove_vals = []
    for idx, row in raw_form.iterrows():
        # Combine PT, SUBJECT, and PG tags in TOC
        if (
            row['SourceClass'] == TextClass.TOC
            and row['tag_list'] == 'PT'
            and row['child_tag_temp'] == 'SUBJECT'
            and row['child_tag_temp_2'] == 'PG'
        ):
            raw_form.at[idx, 'text'] = row['text'] + ' ' + ' ' + row['child_tag_text'] + ' ' + row['child_tag_text_2']
            remove_vals.append((row['child_tag_line_num']))
            remove_vals.append(row['child_tag_line_num_2'])

        # Combine SUBJECT and PG tags in TOC
        if (
            row['SourceClass'] == TextClass.TOC
            and row['tag_list'] == 'SUBJECT'
            and row['child_tag_temp'] == 'PG'
            and row['child_tag_temp_2'] != 'PT'
        ):
            raw_form.at[idx, 'text'] = row['text'] + ' ' + ' ' + row['child_tag_text']
            raw_form.at[idx, 'SourceClass'] = TextClass.HEADING
            remove_vals.append((row['child_tag_line_num']))

    # Process section number entries
    remove_vals_sec = []
    for idx, row in raw_form.iterrows():
        # Combine SECTNO and SUBJECT tags
        if row['tag_list'] == 'SECTNO' and row['child_tag_temp'] == 'SUBJECT':
            raw_form.at[idx, 'text'] = row['text'] + ' ' + ' ' + row['child_tag_text']
            remove_vals_sec.append((row['child_tag_line_num']))

    # Remove processed entries
    raw_form = raw_form[~raw_form['LineNumbers'].isin(remove_vals + remove_vals_sec)]

    # Clean and normalize text
    raw_form.text = raw_form.text.apply(lambda x: str(x).strip())
    raw_form.text = raw_form.text.apply(lambda x: " ".join(x.split()))
    raw_form = raw_form.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace('', np.nan)
    raw_form['xml_idx'] = raw_form.index

    # Keep only important columns
    raw_form = raw_form[['LineNumbers', 'text', 'SourceClass', 'SourceClassName', 'PageNumber', 'xml_idx']]

    # Remove rows with missing data and save to CSV
    raw_form = raw_form.dropna()

    end = time.time()
    log.debug('Time for XML parsing = ' + str(end - start))
    log.debug('SUCCESS')
    return raw_form


def split_paragraphs_into_chunks(text: str, n_words: int) -> list[str]:

    # split the text
    pieces = text.split()

    # return the chunks
    return list(" ".join(pieces[i : i + n_words]) for i in range(0, len(pieces), n_words))


def main() -> None:
    from ai_doc_parser import EASA_SOURCE as xml_path

    xml_path = (
        xml_path.parent
        / "Easy Access Rules for Air Traffic Controllers_ Licensing and Certification _Regulation _EU_ 2015_340_ _PDF_.xml"
    )

    data_dir = Path(__file__).parents[4] / "data" / "documents" / "EASA"
    output_dir = data_dir / "labelled_source"
    output_dir.mkdir(parents=True, exist_ok=True)

    # xml_path = EASA_XML
    log.debug("XML path exists: %s, path: %s", xml_path.exists(), xml_path)

    df = extract_easa_xml(xml_path)
    df.to_csv(output_dir / f"{xml_path.stem}.csv", index=False)
    log.info("Saved to %s", output_dir / f'{xml_path.stem}.csv')


if __name__ == '__main__':
    main()
