import errno
import json
import logging
import os
import sys
import time
import traceback
import zipfile
from datetime import date, datetime

import pandas as pd
from database.document_service import (
    find_document_by_id,
    get_documents_for_parsing_by_type,
    insert_repository_bulk2,
    set_document_as_parsed_pdf,
    set_document_parsed_details,
)
from database.entity.Document import Document
from database.entity.Repository import Repository
from database.entity.ScriptsProperty import ScriptsConfig, parseCredentialFile
from database.utils.MySQLFactory import MySQLDriver
from docx import Document as word_document
from docx2pdf import convert
from lxml import etree as ET

from common_tools.log_config import configure_logging_from_argv
log = logging.getLogger(__name__)

'''
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Console (stdout) handler
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
'''


"""heading calculation"""

def iter_headings(paragraphs):
    for paragraph in paragraphs:
        if paragraph.style.name.startswith("Heading"):
            yield paragraph


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_dir_safe(path):
    if not os.path.exists(path):
        mkdir(path=path)
    return path


def docx_parsing(path: str, path_to_save: str):
    start = time.time()
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
    doc = zipfile.ZipFile(path)
    document_ = doc.open("word/document.xml")
    document_tree = ET.parse(document_)

    """Footnote calculation """
    footnote = doc.open("word/footnotes.xml")
    foornote_tree = ET.parse(footnote)
    footnotes = []
    for elem in foornote_tree.iter():
        foottag = elem.tag.replace(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}", ""
        )
        if foottag == "p":
            for k in elem.iter():
                text = "".join(elem.itertext())
                log.debug("Processing footnote text: %s", text)
                if str(text) != "None" and text.split() and text not in footnotes:
                    # print("here",str(elem.text.split()))
                    # print(k.text)
                    footnotes.append(text)

    """heading calculation"""
    Document_from_docx = word_document(path)
    headings = []
    for heading in iter_headings(Document_from_docx.paragraphs):
        # print(heading.text)
        headings.append(heading.text)

    """table calculation"""
    tables = []
    for table in Document_from_docx.tables:
        for row in table.rows:
            for cell in row.cells:
                tables.append(cell.text)

    """Start Parsing here"""
    for elem in document_tree.iter():
        tag = elem.tag.replace(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}", ""
        )
        """heading calculation"""
        if tag == "t":
            if text in headings and text not in text_list:  # and text not in text_list:
                tag_list.append(tag)
                text_list.append(text)
                Heading.append(1)
                toc.append(0)
                Table.append(0)
                paragraph.append(0)
                page_numbers.append(None)
                Preamble.append(0)
                Footnote_class.append(0)
                Class.append(2)
        """Table Calculation"""
        if tag == "tr":
            for k in elem.iter():
                table_tag = k.tag.replace(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}", ""
                )
                if table_tag == "p":
                    text = "".join(k.itertext())
                    if text.split():
                        # print("Table Text",text)
                        tag_list.append(tag)
                        text_list.append(text)
                        Heading.append(0)
                        toc.append(0)
                        Table.append(1)
                        paragraph.append(0)
                        page_numbers.append(None)
                        Preamble.append(0)
                        Footnote_class.append(0)
                        Class.append(5)
        """TOC calculation"""
        if tag == "sdt":
            for j in elem.iter():
                toc_tag = j.tag.replace(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}", ""
                )
                if toc_tag == "t":
                    text_list.append(j.text)
                    tag_list.append(toc_tag)
                    Heading.append(0)
                    toc.append(1)
                    Table.append(0)
                    paragraph.append(0)
                    page_numbers.append(None)
                    Preamble.append(0)
                    Footnote_class.append(0)
                    Class.append(4)
                    if str(j.text).isnumeric():
                        toc_text = text_list[-2] + "                 " + j.text
                        text_list = text_list[:-1]
                        tag_list = tag_list[:-1]
                        Heading = Heading[:-1]
                        toc = toc[:-1]
                        Table = Table[:-1]
                        paragraph = paragraph[:-1]
                        page_numbers = page_numbers[:-1]
                        Preamble = Preamble[:-1]
                        Footnote_class = Footnote_class[:-1]
                        Class = Class[:-1]
                        text_list[-1] = toc_text
        """Paragraph Calculation"""
        if tag == "p":
            text = "".join(elem.itertext())
            if (
                text not in text_list and "PAGEREF" not in text and text not in headings
            ):  #  and str(text)!='':
                if str(text) != "":
                    tag_list.append(tag)
                    text_list.append(text)
                    Heading.append(0)
                    toc.append(0)
                    Table.append(0)
                    paragraph.append(1)
                    page_numbers.append(None)
                    Preamble.append(0)
                    Footnote_class.append(0)
                    Class.append(0)

    if footnotes:
        for i in footnotes:
            # print("text",i)
            tag_list.append("t")
            text_list.append(i)
            Heading.append(0)
            toc.append(0)
            Table.append(0)
            paragraph.append(0)
            # line_numbers.append(elem.sourceline)
            page_numbers.append(None)
            Preamble.append(0)
            Footnote_class.append(1)
            Class.append(7)
            # print(footnotes)

    taglist = {}
    taglist["tag_list"] = tag_list
    taglist["correct_raw_text"] = text_list
    taglist["header_target"] = Heading
    taglist["paragraph"] = paragraph
    taglist["toc_target"] = toc
    taglist["table_target"] = Table
    taglist["page"] = [1 for i in range(len(tag_list))]
    taglist["Class"] = Class
    log.debug(
        "Parsing results - Tags: %s, Text items: %s, Headings: %s, Tables: %s",
        len(tag_list),
        len(text_list),
        len(Heading),
        len(Table),
    )
    raw_form = pd.DataFrame(taglist)
    path_to_save = (
        path_to_save + "/" + "_" + path.split("/")[-1].strip(".xml") + "xml_xmldf.csv"
    )
    # raw_form.to_csv(path_to_save,index = False)
    end = time.time()
    log.debug("DOCX parsing completed in %s seconds", str(end - start))
    log.debug("DOCX parsing completed successfully")
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
    doc_result: Document = doc
    print("Parsing Docx file:" + doc_result.sourceFileName)

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
    # file_path = doc

    # parser = ET.XMLParser(ns_clean=True)
    # parser = ET.HTMLParser(recover=True)

    try:
        outputList = docx_parsing(file_path, "scripts/")
    except:
        print("Exception >>>>>")
        parseLogText += (
            f"Error in Docx parsing in document: {doc_result.sourceFileName}"
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

        chunks = split_paragraphs_into_chunks(row["correct_raw_text"], 128)
        for chunk in chunks:
            repository = Repository(
                data=chunk,
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


def run(config: ScriptsConfig, docIdsList: int):
    mysql_driver = MySQLDriver(cred=config.databaseConfig.__dict__)
    doc2parse = find_document_by_id(mysql_driver, docIdsList[0])

    document_list = get_documents_for_parsing_by_type(
        mysql_driver, doc2parse.documentType
    )

    docs_dir = config.downloadDir

    if len(docIdsList) > 0:
        document_list2 = []
        for doc in document_list:
            if doc.documentId in docIdsList:
                document_list2.append(doc)
        document_list = document_list2
    print("document list >>> :::" + str(len(document_list)))
    try:
        for doc in document_list:
            if doc.url.endswith(".pdf"):
                srcFileName = os.path.join(config.downloadDir, doc.sourceFileName)
                if os.path.exists(srcFileName):
                    parse(config, mysql_driver, doc)
    except Exception as exc:
        print("^^^^^^ Exception in Docx Parsing" + str(exc))
        traceback.print_exc()

    print("Docx Parser done its job")


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

        configs = parseCredentialFile("/app/tlp_config.json")

        if configs:
            run(configs, docIdsList)
    except Exception as e:
        traceback.print_exc()
        print(e)
