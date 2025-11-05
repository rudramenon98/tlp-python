import json
import logging
import os
import sys
import time
import traceback
from datetime import date, datetime

import lxml.etree as ET
import numpy as np
import pandas as pd
from database.document_service import (
    find_document_by_id,
    get_documents_for_parsing_by_type,
    insert_repository_bulk2,
    set_document_as_parsed,
    set_document_parsed_details,
)
from database.entity.Document import PublicDocument, getDocumentClass
from database.entity.Repository import PublicRepository, getRepositoryClass
from database.entity.ScriptsProperty import ScriptsConfig, parseCredentialFile
from database.utils.MySQLFactory import MySQLDriver

from common_tools.log_config import configure_logging_from_argv
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
'''
log.setLevel(logging.DEBUG)

# Console (stdout) handler
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
'''

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


def calculate_roman_meta_data(df2):
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


def populate_srcline2pageDict(df2, nLinesInFile):
    count = 0
    prevPageSourceline = 0
    prevPageNoText = ""

    for idx, t in enumerate(df2):
        try:
            count += 1
            pagenoText = t.attrib["P"]
            srcLine = t.sourceline
            if pagenoText == "997":
                print()
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
        print()
    textPageNo = srcline2PageDict.get(srcline, None)
    if textPageNo is None:
        print("ERROR: Page Number Not Found for source line:" + str(srcline))
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


def getCorrectPageNumberN(inputtextPageNo, srcline):
    # check if it is roman or integer
    global romanpages, romanlist, romancount, integerlist, totalRomanPages, totalIntegerPages

    if srcline == 55104:
        print()

    textPageNo = srcline2PageDict.get(srcline, None)
    if textPageNo is None:
        print("ERROR: Page Number Not Found for source line:" + str(srcline))
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
    for line in open(filename):
        lines += 1
    return lines


"""XML Parsing Function"""


def XML_Parsing(path, path_to_save):
    """
    ***Important***
    Path : Path of the file
    path_to_save = Save file to particlular direcotory

    * This function will take path as an argument and return the parsed xml file

    """
    start = time.time()

    xmlFileName = path
    parser = ET.XMLParser(recover=True)
    # tree = ET.parse(xmlFileName, parser=parser)
    tree = ET.parse(xmlFileName)
    root = tree.getroot()

    # Count number of lines##
    numLinesinFile = linecount(xmlFileName)

    ##Get page number
    df2 = root.findall(".//PRTPAGE")
    calculate_roman_meta_data2(df2)
    populate_srcline2pageDict(df2, numLinesinFile)

    # df = pd.read_excel('/home/dshah/Inspird-2023-dev/pdf_parsing_new_version/CFR_Tags.xlsx',header=None)
    df = pd.read_excel("./CFR_Tags.xlsx", header=None)
    df2 = df.rename(columns=df.iloc[2]).loc[3:]

    pages = []
    tag_list = []
    text_list = []
    heading = []
    toc = []
    table = []
    paragraph = []
    line_numbers = []
    page_numbers = []
    pageNoText = "i"
    pageNo = 0
    page_numbers2 = []
    Preamble = []
    footnote = []
    Class = []

    # HeadingTags = ['SECTION', 'PART', 'SUBPART', 'SUBCHAP']
    # HeadingTags = [i for i in list(df2['Heading Tags']) if 'nan' not in str(i)]
    # print(HeadingTags)
    HeadingTags = ["PART", "HD", "TITLENUM", "HEADING", "SECTNO", "SUBJECT"]
    UnLikelyHeadingTags = ["APPRO", "CITA", "AUTH"]
    Tags2Skip = list(df2["Non Text tags"])  # .remove('RESERVED')

    # Tags2Skip = ["AC", "ALPHLIST", "APPENDIX", "BMTR", "BTITLE", "CFRDOC", "CFRTITLE", "CHAPTI", "CITE", "CODE",
    #            "CROSSREF", "E", "EDNOTE", "EFFDNOTP", "EXAMPLE", "EXPL", "EXPLA", "EXTRACT", "FAIDS", "FMTR",
    #            "FTNT", "FTREF", "GPH", "GPO", "IPAR", "LSA", "MATH", "NAME", "NOTE", "OENOTICE", "OFFICE", "PART",
    #            "POSITION", "PRTPAGE", "PUB", "PUBYEAR", "RESERVED", "REVTXT", "SIG", "STARS", "SUBCHAP", "SUBJGRP",
    #            "SUBPART", "SUDOCS", "THISTITL", "TITLENO", "TITLEPG", "TOCTAC", "GID", "ALL", "AMDDATE"]

    Tags2Skip = [i for i in Tags2Skip if i not in ["TOCTAC", "FTNT", "EXTRACT"]]
    # print(Tags2Skip)
    # exit()

    combine_tags = ["PT", "SUBJECT", "PG"]
    # preamble_tags = []

    for elem in tree.iter():
        if (
            elem.sourceline in line_numbers
        ):  # duplicate: already handled as part of some higher tag
            continue
        # Auth_child_tags = new_get_children(root)
        # print(Auth_child_tags)

        # exit()
        if elem.tag in Tags2Skip:
            pageTags = elem.find(".//PRTPAGE")
            for sub_tag in elem:
                if sub_tag == "PRTPAGE":
                    if sub_tag.attrib:
                        pageNoText = sub_tag.attrib["P"]
                        # print('PageNoText: ' + pageNoText)
                        pageNo += 1
                        break
            continue

        TOC = 0
        TABLE = 0
        pramble = 0
        # print(elem)
        if elem.tag == "BTITLE":
            print()
        if elem.tag == "PRTPAGE":  ##Here checking page number
            if elem.attrib:

                pageNoText = elem.attrib["P"]
                if pageNoText == "993":
                    print()
                pageNo += 1

            continue
        if elem.tag in HeadingTags:  ##Here checking heading
            # print(elem.text)
            tag_list.append(elem.tag)
            parentTag = elem.getparent()
            heading.append(1)
            paragraph.append(0)
            text_list.append(elem.text.strip())
            pages.append(pageNoText)
            toc.append(TOC)
            table.append(TABLE)
            line_numbers.append(elem.sourceline)
            page_numbers.append(getCorrectPageNumber(pageNoText))
            page_numbers2.append(getCorrectPageNumberN(pageNoText, elem.sourceline))
            Preamble.append(pramble)
            footnote.append(0)
            Class.append(2)
            continue
        if elem.tag in ["TOC", "CFRTOC"]:  ##Here checking TOC
            TOC = 1
            if elem.tag == 31146:
                print()
            for k in elem:
                if k.tag == "PRTPAGE" and k.sourceline not in line_numbers:
                    pageNoText = k.attrib["P"]
                    if pageNoText == "993":
                        print()
                    pageNo += 1
                for j in k:
                    if j.tag == "PRTPAGE" and j.sourceline not in line_numbers:
                        pageNoText = j.attrib["P"]
                        continue
                    tag_list.append(j.tag)
                    if j.tag in HeadingTags:
                        heading.append(1)
                        paragraph.append(0)
                        Class.append(2)
                        TOC = 0
                    elif j.tag == "P":
                        heading.append(0)
                        paragraph.append(1)
                        Class.append(0)
                        TOC = 0
                    else:
                        heading.append(0)
                        paragraph.append(0)
                        Class.append(4)
                        TOC = 1

                    text_list.append("".join(j.itertext()))
                    pages.append(pageNoText)
                    toc.append(TOC)
                    table.append(TABLE)
                    line_numbers.append(j.sourceline)
                    page_numbers.append(getCorrectPageNumber(pageNoText))
                    page_numbers2.append(
                        getCorrectPageNumberN(pageNoText, j.sourceline)
                    )
                    Preamble.append(pramble)
                    footnote.append(0)
                    continue

            continue
        if elem.tag in ["GPOTABLE", "ROW"]:  ##Here checking table
            TABLE = 1
            childs = getChildren_D(elem, [str(elem.tag)])
            for table_ in childs:
                #
                tag_list.append(table_.tag)
                heading.append(0)
                paragraph.append(0)
                text_list.append("".join(table_.itertext()))
                pages.append(pageNoText)
                table.append(TABLE)
                toc.append(TOC)
                line_numbers.append(table_.sourceline)
                page_numbers.append(getCorrectPageNumber(pageNoText))
                page_numbers2.append(
                    getCorrectPageNumberN(pageNoText, table_.sourceline)
                )
                Preamble.append(pramble)
                Class.append(5)
                footnote.append(0)
            continue
        if elem.tag == "P":
            pages.append(pageNoText)
            heading.append(0)
            paragraph.append(1)
            toc.append(TOC)
            table.append(TABLE)
            text_list.append("".join(elem.itertext()))
            tag_list.append(elem.tag)
            line_numbers.append(elem.sourceline)
            page_numbers.append(getCorrectPageNumber(pageNoText))
            page_numbers2.append(getCorrectPageNumberN(pageNoText, elem.sourceline))
            Preamble.append(pramble)
            Class.append(0)
            footnote.append(0)
            continue

        if elem.tag == "E":
            continue

        if elem.tag == "AUTH":
            pramble = 1
            childs = getChildren_D(elem, ["AUTH"])

            for child in childs:
                if (
                    child.tag not in Tags2Skip
                    and child.text
                    and str(child.text) != "nan"
                    and len(child.text.strip()) > 0
                ):
                    pages.append(pageNoText)
                    heading.append(0)
                    paragraph.append(1)
                    toc.append(TOC)
                    table.append(TABLE)
                    text_list.append(child.text.strip())
                    tag_list.append(child.tag)
                    line_numbers.append(child.sourceline)
                    page_numbers.append(getCorrectPageNumber(pageNoText))
                    page_numbers2.append(
                        getCorrectPageNumberN(pageNoText, child.sourceline)
                    )
                    Preamble.append(pramble)
                    Class.append(9)
                    footnote.append(0)
            continue
        if elem.tag == "TOCTAC":
            print("here")
            # exit()
            pramble = 1
            childs = getChildren_D(elem, ["TOCTAC"])

            for child in childs:
                if (
                    child.tag not in Tags2Skip
                    and child.text
                    and str(child.text) != "nan"
                    and len(child.text.strip()) > 0
                ):
                    pages.append(pageNoText)
                    heading.append(0)
                    paragraph.append(1)
                    toc.append(TOC)
                    table.append(TABLE)
                    text_list.append(child.text.strip())
                    tag_list.append(child.tag)
                    line_numbers.append(child.sourceline)
                    page_numbers.append(getCorrectPageNumber(pageNoText))
                    page_numbers2.append(
                        getCorrectPageNumberN(pageNoText, child.sourceline)
                    )
                    Preamble.append(pramble)
                    Class.append(9)
                    footnote.append(0)
            continue

        if elem.tag == "CONTENTS":
            pramble = 1

            childs = getChildren_D(elem, ["CONTENTS"])

            for child in childs:
                if (
                    child.tag not in Tags2Skip
                    and child.text
                    and str(child.text) != "nan"
                    and len(child.text.strip()) > 0
                ):
                    # print(child.text.strip())
                    pages.append(pageNoText)

                    heading.append(0)
                    paragraph.append(1)
                    toc.append(TOC)
                    table.append(TABLE)
                    text_list.append(child.text.strip())
                    tag_list.append(child.tag)
                    line_numbers.append(child.sourceline)
                    page_numbers.append(getCorrectPageNumber(pageNoText))
                    page_numbers2.append(
                        getCorrectPageNumberN(pageNoText, child.sourceline)
                    )
                    Preamble.append(pramble)
                    Class.append(9)
                    footnote.append(0)
            continue
        if elem.tag == "EXTRACT":
            pramble = 0

            childs = getChildren_D(elem, ["EXTRACT"])

            for child in childs:
                if (
                    child.tag not in Tags2Skip
                    and child.text
                    and str(child.text) != "nan"
                    and len(child.text.strip()) > 0
                ):
                    # print(child.text.strip())
                    pages.append(pageNoText)

                    heading.append(1)
                    paragraph.append(0)
                    toc.append(TOC)
                    table.append(TABLE)
                    text_list.append("".join(child.itertext()))
                    tag_list.append(child.tag)
                    line_numbers.append(child.sourceline)
                    page_numbers.append(getCorrectPageNumber(pageNoText))
                    page_numbers2.append(
                        getCorrectPageNumberN(pageNoText, child.sourceline)
                    )
                    Preamble.append(pramble)
                    Class.append(0)
                    footnote.append(0)
            continue

        if elem.tag == "FTNT":
            # print('here')
            # exit()
            # pramble =0
            # childs =getChildren_D(elem, ['FTNT'])
            # print('here')
            # for child in childs:
            #     if child.tag not in Tags2Skip and child.text and str(child.text) !='nan' and len(child.text.strip()) > 0:
            pages.append(pageNoText)
            heading.append(0)
            paragraph.append(1)
            toc.append(TOC)
            table.append(TABLE)
            text_list.append("".join(elem.itertext()))
            tag_list.append(elem.tag)
            line_numbers.append(elem.sourceline)
            page_numbers.append(getCorrectPageNumber(pageNoText))
            page_numbers2.append(getCorrectPageNumberN(pageNoText, elem.sourceline))
            Preamble.append(pramble)
            Class.append(8)
            footnote.append(1)
            continue

        if elem.text and str(elem.text) != "nan" and len(elem.text.strip()) > 0:

            pages.append(pageNoText)

            if elem.tag in HeadingTags:
                heading.append(1)
                paragraph.append(0)
                Class.append(2)
            elif elem.tag in HeadingTags:
                heading.append(0)
                paragraph.append(1)
                Class.append(0)
            else:
                parentTag = elem.getparent()
                if parentTag is not None and parentTag.tag in HeadingTags:
                    heading.append(1)
                    paragraph.append(0)
                    Class.append(2)
                else:
                    heading.append(0)
                    paragraph.append(1)
                    Class.append(0)
            toc.append(TOC)
            table.append(TABLE)
            text_list.append(elem.text)
            tag_list.append(elem.tag)
            line_numbers.append(elem.sourceline)
            Preamble.append(pramble)
            footnote.append(0)
            if len(pageNoText.strip()) > 0:
                page_numbers.append(getCorrectPageNumber(pageNoText))
                page_numbers2.append(getCorrectPageNumberN(pageNoText, elem.sourceline))
            else:
                page_numbers.append(1)
                page_numbers2.append(1)
                #            page_numbers2.append(getCorrectPageNumber2(pageNoText))

    ##Creating DataFrame

    taglist = {}
    taglist["LineNumbers"] = line_numbers
    taglist["tag_list"] = tag_list
    taglist["text"] = text_list
    taglist["heading"] = heading
    taglist["paragraph"] = paragraph
    taglist["TOC"] = toc
    taglist["Table"] = table
    taglist["Pages"] = pages
    # taglist['PageNumber'] = page_numbers
    taglist["PageNumber"] = page_numbers2
    taglist["Preamble"] = [int(i) for i in Preamble]
    taglist["Class"] = Class
    taglist["Footnote"] = footnote
    # print(len(line_numbers),len(tag_list),len(text_list),len(heading),len(paragraph),len(toc),len(table),len(pages),len(page_numbers),len(page_numbers2))

    raw_form = pd.DataFrame(taglist)
    ###Filling Page No
    # raw_form['Pages']  = raw_form['Pages'].ffill()

    ##Cleaning Text

    raw_form["child_tag_temp"] = raw_form["tag_list"].shift(-1)
    raw_form["child_tag_temp_2"] = raw_form["tag_list"].shift(-2)

    raw_form["child_tag_text"] = raw_form["text"].shift(-1)
    raw_form["child_tag_text_2"] = raw_form["text"].shift(-2)

    raw_form["child_tag_line_num"] = raw_form["LineNumbers"].shift(-1)
    raw_form["child_tag_line_num_2"] = raw_form["LineNumbers"].shift(-2)

    remove_vals = []
    for i in raw_form.iterrows():
        if (
            i[1]["TOC"] == 1
            and i[1]["tag_list"] == "PT"
            and i[1]["child_tag_temp"] == "SUBJECT"
            and i[1]["child_tag_temp_2"] == "PG"
        ):
            # df.at[i[0],]
            # print(i[1]['child_tag_text'])
            raw_form.at[i[0], "text"] = (
                i[1]["text"]
                + " "
                + " "
                + i[1]["child_tag_text"]
                + " "
                + i[1]["child_tag_text_2"]
            )
            remove_vals.append((i[1]["child_tag_line_num"]))
            remove_vals.append(i[1]["child_tag_line_num_2"])
            # print(i[1]['child_tag_line_num'],i[1]['child_tag_line_num_2'])
        if (
            i[1]["TOC"] == 1
            and i[1]["tag_list"] == "SUBJECT"
            and i[1]["child_tag_temp"] == "PG"
            and i[1]["child_tag_temp_2"] != "PT"
        ):
            # df.at[i[0],]
            # print(i[1]['child_tag_text'])
            raw_form.at[i[0], "text"] = (
                i[1]["text"] + " " + " " + i[1]["child_tag_text"]
            )
            raw_form.at[i[0], "heading"] = 0
            raw_form.at[i[0], "Class"] = 2

            remove_vals.append((i[1]["child_tag_line_num"]))
            # remove_vals.append(i[1]['child_tag_line_num_2'])
            # print(i[1]['child_tag_line_num'],i[1]['child_tag_line_num_2'])

    remove_vals_sec = []
    for i in raw_form.iterrows():
        if i[1]["tag_list"] == "SECTNO" and i[1]["child_tag_temp"] == "SUBJECT":
            # df.at[i[0],]
            # print(i[1]['child_tag_text'])
            raw_form.at[i[0], "text"] = (
                i[1]["text"] + " " + " " + i[1]["child_tag_text"]
            )
            remove_vals_sec.append((i[1]["child_tag_line_num"]))
            # remove_vals_sec.append(i[1]['child_tag_line_num_2'])

    raw_form = raw_form[~raw_form["LineNumbers"].isin(remove_vals + remove_vals_sec)]

    raw_form.text = raw_form.text.apply(lambda x: str(x).strip())
    raw_form.text = raw_form.text.apply(lambda x: " ".join(x.split()))
    raw_form = raw_form.apply(
        lambda x: x.str.strip() if isinstance(x, str) else x
    ).replace("", np.nan)
    raw_form = raw_form.sort_values(["LineNumbers"], ascending=True)
    raw_form = raw_form.drop(
        [
            "child_tag_temp",
            "child_tag_temp_2",
            "child_tag_text",
            "child_tag_text_2",
            "child_tag_line_num",
            "child_tag_line_num_2",
            "Pages",
            "Footnote",
        ],
        axis=1,
    )
    # path_to_save = '/home/dshah/Inspird-2023-dev/Training_Dataset/updated_cfr_parsing/'+'_'+path.split('/')[-1].strip('.xml')+'xml_xmldf.csv'
    raw_form = raw_form.dropna()
    path_to_save = (
        path_to_save + "/" + "_" + path.split("/")[-1].strip(".xml") + "xml_xmldf.csv"
    )
    # raw_form.to_csv(path_to_save,index = False)
    end = time.time()
    print("Time for XML parsing = " + str(end - start))
    print("SUCCESS")
    return raw_form


def split_paragraphs_into_chunks(text, n_words):

    # split the text
    pieces = text.split()

    # return the chunks
    return list(
        " ".join(pieces[i : i + n_words]) for i in range(0, len(pieces), n_words)
    )


def parse(config: ScriptsConfig, mysql_driver: MySQLDriver, doc):
    # URL = 'https://www.govinfo.gov/content/pkg/CFR-2021-title14-vol4/pdf/CFR-2021-title14-vol4.pdf'
    # doc_result: Document = find_document_by_url(mysql_driver, URL)
    doc_result: PublicDocument = doc
    print("Parsing CFR XML file:" + doc_result.sourceFileName)
    parseLogText = (
        "Document id: "
        + str(doc.documentId)
        + ": "
        + "parsing started at "
        + str(datetime.today().strftime("%d/%m/%Y %H:%M:%S"))
    )
    print(parseLogText)
    set_document_parsed_details(mysql_driver, doc, parseLogText, 0, doc_class=PublicDocument)

    file_path = os.path.join(config.downloadDir, doc_result.sourceFileName)
    # parser = ET.XMLParser(ns_clean=True)
    # parser = ET.HTMLParser(recover=True)

    try:
        outputList = XML_Parsing(file_path, "scripts/")
    except:
        print("Exception >>>>>")
        parseLogText += (
            f"Error in CFR XML parsing in document: {doc_result.sourceFileName}"
        )
        parseLogText += traceback.format_exc()
        print(parseLogText)
        set_document_parsed_details(mysql_driver, doc_result, parseLogText, 0, doc_class=PublicDocument)
        return

    # print(outputList)
    repository_list = []

    for idx, row in outputList.iterrows():
        if not len(row) > 0:
            continue

        chunks = split_paragraphs_into_chunks(row["text"], 128)
        for chunk in chunks:
            repository = PublicRepository(
                data=chunk,
                documentID=doc_result.documentId,
                # pageNoText=row['PageNumber'],
                pageNo=row["PageNumber"],
                # paraNo=idx + 1,
                Type=row["Class"],
                wordCount=len(chunk.split()),
                embedding = 0,
            )
            repository_list.append(repository)

    if len(repository_list) > 0:
        print("Extracted Paragraphs from: " + doc_result.sourceFileName)
        #        delete_repository_data_by_doc_id(mysql_driver, doc_result.documentId)
        insert_repository_bulk2(mysql_driver, repository_list)
        set_document_as_parsed(mysql_driver, doc_result, doc_class=PublicDocument)
        parseLogText += (
            "Successfully parsed document: "
            + doc_result.sourceFileName
            + " and inserted "
            + str(len(repository_list))
            + " paragraphs into the DB"
        )
        print(parseLogText)
        set_document_parsed_details(
            mysql_driver, 
            doc_result, 
            parseLogText, 
            len(repository_list), 
            doc_class=PublicDocument
        )
    else:
        parseLogText += "Error in parsing document: " + doc_result.sourceFileName
        print(parseLogText)
        set_document_parsed_details(mysql_driver, doc_result, parseLogText, 0)


def run(config: ScriptsConfig, docIdsList: int, repo_id:int):
    mysql_driver = MySQLDriver(cred=config.databaseConfig.__dict__)

    doc2parse = find_document_by_id(mysql_driver, docIdsList, doc_class=PublicDocument)

    document_list = get_documents_for_parsing_by_type(mysql_driver, 
                                                      doc2parse.documentType, 
                                                      doc_class=PublicDocument)

    #    document_list = get_documents_for_parsing_by_type(mysql_driver, 1)

    if docIdsList > 0:
        document_list2 = []
        for doc in document_list:
            if doc.documentId == docIdsList:
                document_list2.append(doc)
        document_list = document_list2
    print("document list >>> :::" + str(len(document_list)))
    try:
        for doc in document_list:
            parse(config, mysql_driver, doc)
    except Exception as exc:
        print("^^^^^^ Exception in CFR XML Parsing" + str(exc))
        traceback.print_exc()

    print("XML Parser done its job")

def parse_remaining_args(cleaned_args):
    repo_id = None
    values = []

    i = 0
    while i < len(cleaned_args):
        if cleaned_args[i] == '--repo_id':
            i += 1
            if i >= len(cleaned_args):
                print("Missing value for --repo_id")
            repo_id = int(cleaned_args[i])
        else:
            # handle bracketed list
            if cleaned_args[i].startswith("["):
                list_str = cleaned_args[i]
                while not cleaned_args[i].endswith("]"):
                    i += 1
                    if i >= len(cleaned_args):
                        raise ValueError("Unclosed bracket in list")
                    list_str += " " + cleaned_args[i]
                list_str = list_str.strip("[]")
                values = [int(x) for x in list_str.split()]
            else:
                # handle individual integers outside brackets
                values.append(int(cleaned_args[i]))
        i += 1

    return repo_id, values

if __name__ == "__main__":
    try:
        #configure the logging level
        remaining_args = configure_logging_from_argv(default_level='INFO')
        repo_id, docIdsList = parse_remaining_args(remaining_args)

        if len(docIdsList) > 0:
            scrapeURLId = docIdsList[0]
        else:
            scrapeURLId = 1

        configs = parseCredentialFile("/app/tlp_config.json")

        if configs:
            run(configs, scrapeURLId, repo_id)
    except Exception as e:
        traceback.print_exc()
        print(e)
