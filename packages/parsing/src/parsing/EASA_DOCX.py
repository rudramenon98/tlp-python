import json
import logging
import os
import sys
import time
import traceback
import zipfile
from datetime import date, datetime

import pandas as pd
from clean_text_detect import detect
from common_tools.log_config import configure_logging_from_argv
from database.document_service import (
    find_document_by_id,
    get_documents_for_parsing_by_type,
    insert_repository_bulk2,
    set_document_as_parsed,
    set_document_parsed_details,
)
from database.entity.Document import Document
from database.entity.Repository import Repository
from database import CONFIG_DIR
from database.entity.ScriptsProperty import ScriptsConfig, parseCredentialFile
from database.utils.MySQLFactory import MySQLDriver
from lxml import etree as ET

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
i'''
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Console (stdout) handler
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
'''

def Error_Handler(fn):
    def Inner_Function(*args, **kwargs):
        try:
            t1 = time.time()
            ret = fn(*args, **kwargs)
            t2 = time.time()
            #            print(f'Function {fn.__name__} executed in {(t2-t1):.4f}s')
            return ret
        except Exception as exc:
            log.debug("%s", fn.__name__)
            traceback.print_exc()
            raise

    return Inner_Function


@Error_Handler
def easa_doc_parsing(path, path_to_save):
    start = time.time()
    tree = ET.parse(path)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    tag_list = []
    text_list = []
    Heading = []
    toc = []
    Table = []
    paragraph = []
    page_numbers = []
    Preamble = []
    Footnote_class = []
    Class = []
    toc_text = []
    for i in tree.iter():
        try:
            style = i.find(".//w:pStyle", ns)

            if style is not None:
                tag = i.tag.replace(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}", ""
                )
                check_tag = style.attrib[
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
                ]
                # print(check_tag)
                if (
                    str(check_tag).lower().startswith("heading")
                    or "heading" in str(check_tag).lower()
                ):
                    # tag = i.tag.replace('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}','')
                    if tag == "p":
                        headingext = "".join(i.itertext())
                        if headingext != "" and detect(headingext):
                            # print(''.join(i.itertext()),check_tag)
                            tag_list.append(tag)
                            text_list.append(headingext)
                            Heading.append(1)
                            toc.append(0)
                            Table.append(0)
                            paragraph.append(0)
                            page_numbers.append(None)
                            Preamble.append(0)
                            Footnote_class.append(0)
                            Class.append(2)
                        # continue

                elif str(check_tag).lower().startswith("table"):
                    # tag = i.tag.replace('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}','')
                    if tag == "p":
                        table_text = "".join(i.itertext())
                        if table_text != "" and detect(table_text):
                            tag_list.append(tag)
                            text_list.append(table_text)
                            Heading.append(0)
                            toc.append(0)
                            Table.append(1)
                            paragraph.append(0)
                            page_numbers.append(None)
                            Preamble.append(0)
                            Footnote_class.append(0)
                            Class.append(5)
                        # continue
                        # print(''.join(i.itertext()),check_tag)
                elif str(check_tag).lower().startswith("toc"):
                    # print("tag",tag)
                    if tag == "p":
                        for j in i.iter():
                            # print(j.tag)
                            toctag = j.tag.replace(
                                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}",
                                "",
                            )

                            if toctag == "p":
                                toctext = "".join(j.itertext()).split()
                                toctext = "  ".join(
                                    [
                                        i
                                        for i in toctext
                                        if i.startswith("\\") != True
                                        and "PAGEREF" != i
                                        and i.startswith("_Toc") != True
                                    ]
                                )
                                # print(toctext)
                                # table_text = ''.join(i.itertext())
                                if toctext != "" and detect(toctext):
                                    tag_list.append(tag)
                                    text_list.append(toctext)
                                    Heading.append(0)
                                    toc.append(1)
                                    Table.append(0)
                                    paragraph.append(0)
                                    page_numbers.append(None)
                                    Preamble.append(0)
                                    Footnote_class.append(0)
                                    Class.append(4)
                                # continue
                else:
                    if tag == "p":
                        paragraphtext = "".join(i.itertext())

                        # if paragraphtext not in text_list:
                        if str(paragraphtext) != "":  # and  str(paragraphtext):
                            # print("Text", paragraphtext)

                            if detect(paragraphtext):
                                tag_list.append(tag)
                                text_list.append(paragraphtext)
                                Heading.append(0)
                                toc.append(0)
                                Table.append(0)
                                paragraph.append(1)
                                page_numbers.append(None)
                                Preamble.append(0)
                                Footnote_class.append(0)
                                Class.append(0)

            else:
                # print(i.tag)
                tag = i.tag.replace(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}", ""
                )
                if tag == "p":
                    paragraphtext = "".join(i.itertext())

                    # if paragraphtext not in text_list:
                    if paragraphtext != "":
                        if detect(paragraphtext):
                            print("paraptext", paragraphtext)
                            tag_list.append(tag)
                            text_list.append(paragraphtext)
                            Heading.append(0)
                            toc.append(0)
                            Table.append(0)
                            paragraph.append(1)
                            page_numbers.append(None)
                            Preamble.append(0)
                            Footnote_class.append(0)
                            Class.append(0)

        except:
            continue

    taglist = {}
    taglist["tag_list"] = tag_list
    taglist["text"] = text_list
    taglist["heading"] = Heading
    taglist["paragraph"] = paragraph
    taglist["TOC"] = toc
    taglist["Table"] = Table
    taglist["Pages"] = [1 for i in range(len(tag_list))]

    taglist["Class"] = Class

    raw_form = pd.DataFrame(taglist)
    # raw_form = raw_form[7:]
    raw_form = raw_form.dropna()
    path_to_save = (
        path_to_save + "/" + "_" + path.split("/")[-1].strip(".xml") + "xml_xmldf.csv"
    )
    end = time.time()
    print("Time for EASA XML parsing = " + str(end - start))
    print("SUCCESS")
    return raw_form


# @Error_Handler
def split_paragraphs_into_chunks(text, n_words):

    # split the text
    pieces = text.split()

    # return the chunks
    return list(
        " ".join(pieces[i : i + n_words]) for i in range(0, len(pieces), n_words)
    )


@Error_Handler
def parse(config: ScriptsConfig, mysql_driver: MySQLDriver, doc):
    doc_result: Document = doc
    print("Parsing EASA XML file:" + doc_result.sourceFileName)
    parseLogText = (
        "Document id: "
        + str(doc.documentId)
        + ": "
        + "parsing started at "
        + str(datetime.today().strftime("%d/%m/%Y %H:%M:%S"))
    )
    print(parseLogText)
    set_document_parsed_details(mysql_driver, doc, parseLogText, 0)
    file_path = os.path.join(config.downloadDir, doc_result.sourceFileName)

    ### Longest common prfix with case
    ##sort igonre case

    try:
        outputList = easa_doc_parsing(file_path, "scripts/")
    except:
        print("Exception >>>>>")
        parseLogText += (
            f"Error in EASA XML parsing in document: {doc_result.sourceFileName}"
        )
        parseLogText += traceback.format_exc()
        print(parseLogText)
        set_document_parsed_details(mysql_driver, doc_result, parseLogText, 0)
        return

    repository_list = []

    for idx, row in outputList.iterrows():
        if not len(row) > 0:
            continue

        chunks = split_paragraphs_into_chunks(row["text"], 128)

        for chunk in chunks:
            paraText = chunk.encode("utf-8").decode("utf-8")
            paraText2 = "".join(x for x in paraText if ord(x) < 65536)

            repository = Repository(
                data=paraText2,
                documentID=doc_result.documentId,
                # pageNoText=row['PageNumber'],
                pageNo=0,
                # paraNo=idx + 1,
                Type=row["Class"],
                wordCount=len(chunk.split()),
            )
            repository_list.append(repository)

    if len(repository_list) > 0:
        print("Extracted Paragraphs from: " + doc_result.sourceFileName)
        #        delete_repository_data_by_doc_id(mysql_driver, doc_result.documentId)
        insert_repository_bulk2(mysql_driver, repository_list)
        set_document_as_parsed(mysql_driver, doc_result)
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


@Error_Handler
def run(config: ScriptsConfig, docIdsList: int):
    mysql_driver = MySQLDriver(cred=config.databaseConfig.__dict__)
    doc2parse = find_document_by_id(mysql_driver, docIdsList[0])

    document_list = get_documents_for_parsing_by_type(
        mysql_driver, doc2parse.documentType
    )

    #    document_list = get_documents_for_parsing_by_type(mysql_driver, 6)
    for i in document_list:
        print(i.url)

    if len(docIdsList) > 0:
        document_list2 = []
        for doc in document_list:
            if doc.documentId in docIdsList:
                document_list2.append(doc)
        document_list = document_list2
    print("document list >>> :::" + str(len(document_list)))
    try:
        for doc in document_list:
            parse(config, mysql_driver, doc)
    except Exception as exc:
        print("^^^^^^ Exception in EASA XML Parsing" + str(exc))
        traceback.print_exc()

    print("EASA Parser done its job")


if __name__ == "__main__":
    try:
        #configure the logging level
        remaining_args = configure_logging_from_argv(default_level='INFO')

        docIdsList = []

        if len(remaining_args) >= 1:
            n = len(remaining_args[0])
            docs = remaining_args[0][1 : n - 1]
            docs = docs.split(" ")
            docIdsList = [int(i) for i in docs]

        configs = parseCredentialFile(str(CONFIG_DIR / "dev_test_tlp_config.json"))

        if configs:
            run(configs, docIdsList)
    except Exception as e:
        traceback.print_exc()
        print(e)
