import logging
import os
import re
import sys
import time
import traceback

import lxml
import pandas as pd
from common_tools.log_config import configure_logging_from_argv
from database.document_service import (
    find_document_by_id,
    get_documents_for_parsing_by_type,
    insert_repository_bulk2,
    set_document_as_parsed_pdf,
    set_document_parsed_details,
)
from database.entity.Document import Document
from database.entity.Repository import Repository
from database import CONFIG_DIR
from database.entity.ScriptsProperty import ScriptsConfig, parseCredentialFile
from database.utils.MySQLFactory import MySQLDriver

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Console (stdout) handler
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)


def First_word_iscompound(x):
    first_word_iscompound = 0
    temp_text = x
    # is_compond_pattern = '^([a-zA-Z0-9](?:\.[a-zA-Z0-9]+)?)(?:\s)'
    # is_compond_pattern = r"^(([$&+:=?@#|'<>.-^*()%!{}-][\d]+|([$&+:=?@#|'<>.-^*()%!{}-][a-zA-Z]))|[\d]+|[a-zA-Z])((\.[0-9a-zA-Z]+)+)?(?:[$&+:=?@#|'<>.-^*()%!{}-]?\s)"
    # is_compond_pattern = r"^(([+()%!-^{}-][\d]+|([()+%!-^{}-][a-zA-Z]))|[\d]+|[a-zA-Z])((\.[0-9a-zA-Z]+)+)?(?:[()+%![\]{}-]?\s)"
    global is_compond_pattern

    is_compond_pattern = r"^(([+()%!-^{}-•.][\d]+|([()+%!-^{}-•.][a-zA-Z]+))|[\d]+|[a-zA-Z])((\.[0-9a-zA-Z]+)+)?(?:[()+%![\]{}-•.]?\s)"
    is_roman_pattern = r"^M{0,3}(CM|CD|D?C{0,3})?(XC|XL|L?X{0,3})?(IX|IV|V?I{0,3})?$\."
    first_char_pattern = r"^[A-Za-z\d]+(?:\s)"
    first_char = re.match(first_char_pattern, temp_text)
    is_compound_ = re.findall(is_compond_pattern, temp_text)
    Is_Compound_roman = re.match(is_roman_pattern, temp_text)
    if (
        is_compound_ and not first_char and not temp_text.isdigit()
    ) or Is_Compound_roman:
        first_word_iscompound = 1

    # text = x
    # if functions.is_compound_(text.split()[0].strip()):#text.split()[0].count(".")>0 or compound_detector(text.split()[0]):
    #     first_word_iscompound = 1
    return first_word_iscompound


def HTML_Parsing(file_path: str):
    start = time.time()

    with open(file_path, "r") as f1:
        tree = lxml.html.parse(f1)

    ###html parsing

    headers = ["h" + str(i) for i in range(1, 10)]
    tag_list = []
    text_list = []
    Class = []
    line_numbers = []
    for i in tree.iter():
        try:
            if i.tag == "p":
                if bool(i.attrib):
                    if str(i.attrib["class"]).lower() == "footnote":
                        # print('here',''.join(i.itertext()))
                        text = " ".join(i.itertext()).replace("\n", "")
                        text = text.strip()
                        text = re.sub(
                            r'(\s([?,.!"]))|(?<=\[|\()(.*?)(?=\)|\])',
                            lambda x: x.group().strip(),
                            text,
                        )
                        # print(text)
                        compound = First_word_iscompound(text)
                        if compound:
                            text = re.sub(r"\s", " ", text)
                            text = text.split("   ")
                            for new_text in text:
                                text_list.append(new_text)
                                Class.append(8)
                                line_numbers.append(lineumber)
                                tag_list.append("Footnote")
                        else:
                            #                             if First_word_iscompound(text+' '):
                            #                                 continue
                            Class.append(8)

                            tag_list.append("Footnote")
                            text_list.append(text)

                            line_numbers.append(lineumber)
                    else:

                        lineumber = i.sourceline
                        text = " ".join(i.itertext()).replace("\n", "")

                        compound = First_word_iscompound(text)
                        if compound:
                            text = re.sub(r"\s", " ", text)
                            text = text.split("   ")
                            for new_text in text:
                                # print("new text",new_text)
                                text_list.append(new_text)
                                Class.append(0)
                                line_numbers.append(lineumber)
                                tag_list.append(i.tag)
                                # text =re.split(is_compond_pattern,text)

                        else:
                            # print("not compund text",text)
                            text_list.append(text)
                            Class.append(0)
                            line_numbers.append(lineumber)
                            tag_list.append(i.tag)
            if i.tag in headers:
                # print(i.tag)
                lineumber = i.sourceline
                text = " ".join(i.itertext()).replace("\n", "")
                text = text.strip()
                text = re.sub(
                    r'(\s([?,.!"]))|(?<=\[|\()(.*?)(?=\)|\])',
                    lambda x: x.group().strip(),
                    text,
                )
                # print(text)
                compound = First_word_iscompound(text)
                if compound:
                    text = re.sub(r"\s", " ", text)
                    text = text.split("   ")
                    for new_text in text:
                        text_list.append(new_text)
                        Class.append(2)
                        line_numbers.append(lineumber)
                        tag_list.append("HD")
                else:
                    Class.append(2)

                    tag_list.append("HD")
                    text_list.append(text)
                    line_numbers.append(lineumber)
            if i.tag.startswith("table"):
                Class.append(5)
                tag_list.append("TABLE")
                text_list.append(text)
                line_numbers.append(lineumber)
        except:
            continue
    taglist = pd.DataFrame()
    log.debug(
        "Parsing results - Line numbers: %s, Tags: %s, Text items: %s, Classes: %s",
        len(line_numbers),
        len(tag_list),
        len(text_list),
        len(Class),
    )
    taglist["LineNumbers"] = line_numbers
    taglist["tag_list"] = tag_list
    taglist["text"] = text_list
    taglist["Class"] = Class
    taglist = taglist.drop_duplicates(subset=["text"], keep="first")
    # print(taglist)
    end = time.time()
    # taglist.to_csv('/home/ravindra/scann-trials/etl-v5/candana_tml.csv',index= False)
    log.debug("HTML parsing completed in %s seconds", str(end - start))
    log.debug("HTML parsing completed successfully")

    return taglist


def split_paragraphs_into_chunks(text, n_words):

    # split the text
    pieces = text.split()

    # return the chunks
    return list(
        " ".join(pieces[i : i + n_words]) for i in range(0, len(pieces), n_words)
    )


def parse(config: ScriptsConfig, mysql_driver: MySQLDriver, doc, scrapeURLId):
    # URL = 'https://www.govinfo.gov/content/pkg/CFR-2021-title14-vol4/pdf/CFR-2021-title14-vol4.pdf'
    # doc_result: Document = find_document_by_url(mysql_driver, URL)
    doc_result: Document = doc
    print("Parsing MDR HTML file:" + doc_result.sourceFileName)
    file_path = os.path.join(config.downloadDir, doc_result.sourceFileName)
    parseLogText = ""
    try:
        outputList = HTML_Parsing(file_path)
    except:
        print("Exception >>>>>")
        parseLogText += (
            f"Error in CFR XML parsing in document: {doc_result.sourceFileName}"
        )
        parseLogText += traceback.format_exc()
        print(parseLogText)
        set_document_parsed_details(mysql_driver, doc_result, parseLogText, 0)
        return

    # print(outputList)
    repository_list = []

    for idx, row in outputList.iterrows():
        if not len(row) > 0:
            continue

        chunks = split_paragraphs_into_chunks(row["text"], 128)

        for chunk in chunks:
            repository = Repository(
                # data=row['text'],
                data=chunk,
                pageNo=0,
                documentID=doc_result.documentId,
                Type=row["Class"],
                wordCount=len(chunk.split()),
            )

            repository_list.append(repository)

    if len(repository_list) > 0:
        print("Extracted Paragraphs from: " + doc_result.sourceFileName)
        #        delete_repository_data_by_doc_id(mysql_driver, doc_result.documentId)
        insert_repository_bulk2(mysql_driver, repository_list)
        set_document_as_parsed_pdf(mysql_driver, doc_result)
        parseLogText += (
            "Successfully parsed document: "
            + doc_result.sourceFileName
            + " and inserted "
            + str(len(repository_list))
            + " paragraphs into the DB"
        )
        print(parseLogText)
        set_document_parsed_details(
            mysql_driver, doc_result, parseLogText, len(repository_list)
        )
    else:
        parseLogText += "Error in parsing document: " + doc_result.sourceFileName
        print(parseLogText)
        set_document_parsed_details(mysql_driver, doc_result, parseLogText, 0)


def run(config: ScriptsConfig, docIdsList: int, scrapeURLId):
    mysql_driver = MySQLDriver(cred=config.main_databaseConfig.__dict__)
    doc2parse = find_document_by_id(mysql_driver, docIdsList[0])

    document_list = get_documents_for_parsing_by_type(
        mysql_driver, doc2parse.documentType
    )
    # docIdsList = [692]

    if len(docIdsList) > 0:
        document_list2 = []
        for doc in document_list:
            print(doc.documentId)
            if doc.documentId in docIdsList:
                document_list2.append(doc)
        document_list = document_list2
    try:
        for doc in document_list:
            parse(config, mysql_driver, doc, scrapeURLId)
            """hard coded"""
    except Exception:
        print("Exception in MDR XML Parsing")
        traceback.print_exc()


if __name__ == "__main__":
    try:
        # configure the logging level
        remaining_args = configure_logging_from_argv(default_level="INFO")

        docIdsList = []
        n = len(remaining_args[0])
        docs = remaining_args[0][1 : n - 1]
        docs = docs.split(" ")
        docIdsList = [int(i) for i in docs]

        configs = parseCredentialFile(str(CONFIG_DIR / "dev_test_tlp_config.json"))

        if configs:
            run(configs, docIdsList, 1)

    except Exception as e:
        traceback.print_exc()
        print(e)
