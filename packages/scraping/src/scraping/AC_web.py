## Library import
import json
import logging
import os
import time
import traceback
from datetime import datetime
from multiprocessing import current_process

import pandas as pd
import requests
from bs4 import BeautifulSoup
from common_tools.log_config import configure_logging_from_argv
from database.document_service import (
    cancel_documents,
    find_document_by_url,
    find_documents_not_scraped_on_date,
    get_scrape_script_by_scraperUrlId,
    insert_documents_bulk2,
    update_documents,
)
from database.entity.Document import Document
from database.entity.ScrapScript import ScrapScript
from database.entity.ScriptsProperty import ScriptsConfig, parseCredentialFile
from database.scrape_url_service import (
    scrape_url_append_log,
    update_scrape_url_set_log_value,
)
from database.utils.MySQLFactory import MySQLDriver
from database.utils.util import get_dir_safe

log = logging.getLogger(__name__)
"""
log.setLevel(logging.DEBUG)

# Console (stdout) handler
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
"""

# for logging
logList = []
DateToday = datetime.today()


def download_file_low(URL, path):
    hdr = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9,gu;q=0.8,hi;q=0.7",
        "Connection": "keep-alive",
    }
    chunk_size = (1024 * 1024) * 1  # 1MB

    response = requests.get(URL, headers=hdr, stream=True)
    int(response.headers.get("content-length", 0))

    with open(path, "wb") as file:
        for data in response.iter_content(chunk_size=chunk_size):
            file.write(data)


def download_file(config: ScriptsConfig, document: Document, sequence):
    try:
        # mysql_driver = MySQLDriver(cred=config.main_databaseConfig.__dict__)
        file_name = document.title.replace("/", " ") + ".pdf"
        # dir = config.rootDataDir + config.downloadDirInRootDir
        dirpath = config.downloadDir
        path = get_dir_safe(dirpath) + "/" + file_name
        log.debug(
            "%s -> Downloading file %s url:%s on location:%s",
            current_process(),
            sequence,
            document.url,
            path,
        )
        time.sleep(10)
        # response = urllib.request.urlopen(document.url)
        # file = open(path, 'wb')
        # file.write(response.read())
        # file.close()
        download_file_low(document.url, path)
        document.pdfFileName = os.path.relpath(path, config.downloadDir)
        document.fileSize = os.path.getsize(path)
        document.parsed = False
        # insert_document(mysql_driver, document)
        # mysql_driver.close()
        return document, "Success"
    except Exception as ex:
        log.debug("File is not Downloaded: %s", document.title + ".pdf")
        log.debug("%s", ex)
        traceback.print_exc()
        return None, str(traceback.format_exc())


def process_page(config: ScriptsConfig, dfs, mysql_driver, scrapeURLId):
    global logList, DateToday
    cnt = 0

    log.debug("Processing website batch")
    scrapeDF = dfs
    scrapeDF["ID"] = [str(i) for i in range(len(scrapeDF))]
    scrapeDF["ScrapingLog"] = ""

    scrapeDF[["description", "Number"]] = scrapeDF[["description", "Number"]].fillna("")
    # scrapeDF.fillna({'description': '', 'Number':''}, inplace=True)

    mysql_driver2 = MySQLDriver(cred=config.databaseConfig.__dict__)
    # scrapeScript: ScrapScript = get_scrap_script_by_file_name(mysql_driver2, os.path.basename(__file__))
    scrapeScript: ScrapScript = get_scrape_script_by_scraperUrlId(
        mysql_driver2, scrapeURLId
    )

    log.debug("Checking for new documents")
    update_list, download_list = check_for_new_documents(
        mysql_driver2, scrapeDF, scrapeURLId, scrapeScript
    )

    logText = f"Number of new documents  since last scrape: {len(download_list)}"
    logList.append(logText)
    scrape_url_append_log(mysql_driver2, scrapeURLId, logText)
    log.debug("%s", logText)

    logText = f"Number of not updated documents since last scrape: {len(update_list)}"
    logList.append(logText)
    scrape_url_append_log(mysql_driver2, scrapeURLId, logText)
    log.debug("%s", logText)

    log.debug("Updating documents in DB")
    if update_list and len(update_list) > 0:
        update_documents(mysql_driver2, update_list)

    # num_process  =10
    # pool = Pool(num_process)

    # download_queue = []
    log.debug("Final download list size:")
    log.debug("Download List: %s", len(download_list))
    # exit()

    # New Code
    bulk_document = []
    logText = f"File downloading started at {datetime.today()}"
    logList.append(logText)
    t1 = time.time()
    log.debug("Starting download files")
    # cnt = 0
    for doc in download_list:
        log.debug("DOC: %s", str(doc.number))
        cnt += 1
        docs, msgText = download_file(config, doc, cnt)
        #            if docs is None:
        #                continue
        # bulk_document.append(docs)
        if docs is not None:
            bulk_document.append(docs)
        else:
            logList.append(msgText)
            scrape_url_append_log(mysql_driver2, scrapeURLId, msgText)

        # exit()
        # download_queue.append((config, doc, cnt))

    # logText = f'File downloading started at {datetime.today()}'
    # logList.append(logText)
    scrape_url_append_log(mysql_driver2, scrapeURLId, logText)
    log.debug("%s", logText)

    # t1 = time.time()
    # print("starting download files")
    # pool.starmap(download_file, download_queue)
    # print("Download completed in  :  " + str((time.time() - t1)))

    logText = f"File downloading completed at {datetime.today()} time taken: {str((time.time() - t1))}"
    logList.append(logText)
    scrape_url_append_log(mysql_driver2, scrapeURLId, logText)
    log.debug("%s", logText)

    docIDsList = None
    try:
        if bulk_document is not None and len(bulk_document) > 0:
            docIDsList = insert_documents_bulk2(mysql_driver2, bulk_document)

        if docIDsList is not None and len(docIDsList) > 0:
            log.debug("Newly added docIDs: %s", docIDsList)
            logText = f"Newly added docIDs: {docIDsList}"
            logList.append(logText)
            scrape_url_append_log(mysql_driver2, scrapeURLId, logText)
    except:
        logText = traceback.format_exc()
        logList.append(logText)
        scrape_url_append_log(mysql_driver2, scrapeURLId, logText)
        log.debug("%s", logText)

    log.debug("Newly added docIDs: %s", docIDsList)

    # handle cancelled documents
    cancel_list = check_for_cancelled_documents(mysql_driver2, DateToday, scrapeURLId)

    logText = f"Number of cancelled documents since last scrape: {len(cancel_list)}"
    logList.append(logText)
    scrape_url_append_log(mysql_driver2, scrapeURLId, logText)
    log.debug("%s", logText)

    # Mark the newly cancelled documents in the documents table
    if cancel_list and len(cancel_list) > 0:
        cancel_documents(mysql_driver2, cancel_list)

    mysql_driver2.close()
    log.debug("Processed website page successfully")


def check_for_new_documents(
    mysql_driver, scrapeDF, scrapeURLId, scrapeScript: ScrapScript
):
    download_list = []
    update_list = []
    skip_list = []

    try:
        # scrapeScript: ScrapScript = get_scrap_script_by_file_name(mysql_driver, os.path.basename(__file__))
        log.debug("Processing document type ID: %s", scrapeScript.documentTypeID)
    except Exception:
        logText = f"Get ScrapeSCript Failed for : {os.path.basename(__file__)} \n"
        logText += traceback.format_exc()
        logList.append(logText)
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)
        log.debug("%s", logText)
        return update_list, download_list

    for idx, row in scrapeDF.iterrows():

        file_url = row["pdf_file_url"]

        try:
            if len(row["Number"].strip()) == 0:
                continue

            if isinstance(file_url, list):
                if len(file_url) > 0:
                    file_url = str(file_url[0])
                else:
                    skip_list.append(row["Number"] + " " + row["title"])
                    logText = (
                        "Skipped File "
                        + row["title"]
                        + " with AC at "
                        + row["Number"]
                        + " URL = <null>  on "
                        + str(datetime.today())
                    )
                    logList.append(logText)
                    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                    log.debug("%s", logText)
                    continue

            docInDB = find_document_by_url(mysql_driver, file_url)
        except Exception:
            logText = f"Failed for : {file_url} \n"
            logText += traceback.format_exc()
            logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            log.debug("%s", logText)
            continue

        if not docInDB:
            try:
                document = Document(
                    number=row["Number"],
                    title=row["Number"] + " : " + row["title"],
                    description=row["description"][:1024],
                    url=row["pdf_file_url"],
                    # documentType=row['filetype'],
                    documentType=scrapeScript.documentTypeID,  # need to do some changes according to ID
                    documentStatus=1,  # Active
                    activeDate=row["active_date"],
                    inactiveDate=None,
                    sourceFileName=None,
                    pdfFileName=None,
                    parsed=False,
                    embedded=False,
                    parsingScriptID=scrapeScript.defaultParsingScriptID,
                    # parsingScriptID = None,
                    createdDate=datetime.today().date(),
                    modifiedDate=datetime.today().date(),
                    #                additionalInfo=row['pdf_file_name'],
                    fileSize=0,
                    parsingLog="notParsed",
                    embeddingLog="notEmbedded",
                    noOfParagraphs=0,
                    lastScrapeDate=datetime.today().date(),
                    sourceProject=0,
                )
                download_list.append(document)
            except Exception:
                logText = f"New Document row creation failed for : {file_url} \n"
                logText += traceback.format_exc()
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                log.debug("%s", logText)

        else:
            docInDB.lastScrapeDate = datetime.today().date()
            # docInDB.scrapingLog = 'File ' + row['title']  + ' with AC at ' + row['pdf_file_url'] + ' scraped and not changed on ' + str(datetime.today())
            logText = (
                "File "
                + row["title"]
                + " with AC at "
                + row["pdf_file_url"]
                + " accessed and not changed on "
                + str(datetime.today())
            )
            logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            log.debug("%s", logText)
            update_list.append(docInDB)

    return update_list, download_list


def getCancelledACs():
    CancelledACsCSVFileURL = "https://www.faa.gov/regulations_policies/advisory_circulars/index.cfm/go/document.exportAll/statusID/3"
    cancelledACs = pd.read_csv(CancelledACsCSVFileURL)
    return cancelledACs.to_dict("records")


def check_for_cancelled_documents(mysql_driver, current_date, scrapeURLId):
    global logList

    cancel_list = []

    # get the list of cancelled documents from the FAA website
    cancelled_documents_list = getCancelledACs()

    if not cancelled_documents_list or len(cancelled_documents_list) == 0:
        return None

    # get the documents in documents table which were not scraped today
    old_documents_list = find_documents_not_scraped_on_date(mysql_driver, current_date)

    if not old_documents_list or len(old_documents_list) == 0:
        return None

    for cancelled_doc in cancelled_documents_list:
        cancelled_number = cancelled_doc["DOCUMENTNUMBER"]

        for old_doc in old_documents_list:
            old_number = old_doc.number

            if old_number == cancelled_number:
                # Mark as Cancelled/Archived
                old_doc.documentStatus = 3  # Cancelled
                old_doc.inactiveDate = datetime.today().date()
                old_doc.modifiedDate = datetime.today().date()

                # write to log
                logText = (
                    "File "
                    + " ["
                    + str(old_number)
                    + "] "
                    + old_doc.pdfFileName
                    + " with "
                    + str(old_doc.documentType)
                    + " changed to inactive on "
                    + str(str(datetime.today()))
                )
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                log.debug("%s", logText)

                # append to cancel list
                cancel_list.append(old_doc)

    return cancel_list


def save_log_data_to_db(log_list, mysql_driver):
    update_scrape_url_set_log_value(mysql_driver, 2, json.dumps(log_list))


def page_extraction2(mysql_driver, scrapeURLId, url):
    hdr = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9,gu;q=0.8,hi;q=0.7",
        "Connection": "keep-alive",
    }

    try:

        # response = requests.post(URL2, headers=hdr, json=data, timeout=10)
        response2 = requests.get(url, headers=hdr, timeout=10)
        response2.raise_for_status()
    except:
        # traceback.print_exc()
        logText = f"Failed to open url: {url} error={traceback.format_exc()}"
        logList.append(logText)
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)
        log.debug("%s", logText)

    try:
        soup2 = BeautifulSoup(response2.content, "html.parser")
        links = soup2.find_all("article")

        link_row = {
            "Number": "",
            "title": "",
            "description": "",
            "pdf_file_name": "",
            "pdf_file_url": "",
            "filetype": "",
            "status": "",
            "active_date": "",
            "Download": "",
        }

        info = links[0].contents[3].contents

        info_length = len(info)
        for idx in range(info_length):
            if info[idx] == "\n":
                continue
            if idx + 2 >= info_length:
                break
            # print(f"idx={idx} => {info[idx]}")

            if info[idx].contents[0] == "Number" and idx + 2 <= info_length:
                link_row["Number"] = str(info[idx + 2].contents[0])
            elif info[idx].contents[0] == "Title" and idx + 2 <= info_length:
                link_row["title"] = str(info[idx + 2].contents[0])
            elif info[idx].contents[0] == "Status" and idx + 2 <= info_length:
                link_row["status"] = str(info[idx + 2].contents[0])
            elif info[idx].contents[0] == "Date issued" and idx + 2 <= info_length:
                link_row["active_date"] = datetime.strptime(
                    str(info[idx + 2].contents[0]), "%Y-%m-%d"
                ).date()
            elif info[idx].contents[0] == "Description" and idx + 2 <= info_length:
                link_row["description"] = str(info[idx + 2].contents[0])
            elif info[idx].contents[0] == "Content" and idx + 2 <= info_length:
                for a in info[idx + 2].find_all("a"):
                    h_ref = a.attrs["href"]
                    if h_ref.lower().endswith(".pdf"):
                        link_row["pdf_file_url"] = str(h_ref)
                        break
        link_row["Download"] = False
        logText = (
            "Located AC : "
            + " Number: "
            + link_row["Number"]
            + " Title:"
            + link_row["title"]
            + " PDF URL: "
            + str(link_row["pdf_file_url"])
            + "  on "
            + str(datetime.today())
        )
        logList.append(logText)
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)
        log.debug("%s", logText)

        return link_row

    except:
        # traceback.print_exc()
        logText = f"Failed to get details of : {url} error={traceback.format_exc()}"
        logList.append(logText)
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)
        log.debug("%s", logText)

        return None


def get_main_page2(mysql_driver, scrapeURLId, main_url):

    hdr = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9,gu;q=0.8,hi;q=0.7",
        "Connection": "keep-alive",
    }

    try:

        # response = requests.post(URL2, headers=hdr, json=data, timeout=10)
        response = requests.get(main_url, headers=hdr, timeout=10)
        response.raise_for_status()
    except:
        # traceback.print_exc()
        logText = f"Failed to open url: {main_url} error={traceback.format_exc()}"
        logList.append(logText)
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)
        log.debug("%s", logText)

    # print(response.text)

    result = []

    try:
        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find_all("table")
        # table = table[0]
        # for i in table.find_all('a'):
        list_of_links = table[0].find_all("a")
        list_of_ACs = [j for j in list_of_links if "documentID" in j.attrs["href"]]
        count = 0
        # for i in range(0, len(list_of_ACs), 10):
        for i in list_of_ACs:
            count += 1
            if count % 30 == 0:
                time.sleep(10)

            href = i.attrs["href"]
            if "documentID" in href:
                # print (href)
                row = page_extraction2(
                    mysql_driver, scrapeURLId, "https://www.faa.gov/" + href
                )
                if row:
                    result.append(row)
    except:
        # traceback.print_exc()
        logText = f"Failed to get details of Advisory Circulars: {count} error={traceback.format_exc()}"
        logList.append(logText)
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)
        log.debug("%s", logText)

    return result


def scrape_faa_acs2(config: ScriptsConfig, driver, result, mysql_driver, scrapeURLId):
    # dfs = []
    try:

        batch_size = 10
        for i in range(0, len(result), batch_size):
            # print(l[i:i+batch_size])

            batch_result = result[i : i + batch_size]

            df = pd.DataFrame(batch_result)

            log.debug("Processing batch #%s", str(i))

            process_page(config, df, mysql_driver, scrapeURLId)

            time.sleep(10)
    except Exception:
        logText = traceback.format_exc()
        logList.append(logText)
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)
        log.debug("%s", logText)

    # return dfs


def run(config: ScriptsConfig, scrapeURLId):
    global logList
    datetime.today()

    main_URL = "https://www.faa.gov/regulations_policies/advisory_circulars/index.cfm/go/document.list/"

    mysql_driver = MySQLDriver(cred=config.databaseConfig.__dict__)

    logText = f"Scrape for FAA ACs started at {datetime.today()} URL: {main_URL}"
    logList.append(logText)

    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    log.debug("%s", logText)

    result = get_main_page2(mysql_driver, scrapeURLId, main_URL)

    logText = f"#Advisory Circulars: {len(result)}"
    logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    log.debug("%s", logText)

    logText = f"Website data captured at {datetime.today()}"
    logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)

    scrape_faa_acs2(config, mysql_driver, result, mysql_driver, scrapeURLId)

    logText = f"Scraping URL: {main_URL} DONE"
    logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    log.debug("%s", logText)

    # save_log_data_to_db(logList, mysql_driver)
    log.debug("Scraped URL: %s", main_URL)


if __name__ == "__main__":
    try:
        props = None

        # configure the logging level
        remaining_args = configure_logging_from_argv(default_level="INFO")

        docIdsList = []
        if len(remaining_args) >= 1:
            n = len(remaining_args[0])
            docs = remaining_args[0][1 : n - 1]
            docs = docs.split(" ")
            docIdsList = [int(i) for i in docs]

        if len(docIdsList) > 0:
            scrapeURLId = docIdsList[0]
        else:
            scrapeURLId = 2

        configs = parseCredentialFile("/app/tlp_config.json")
        #        configs = parseCredentialFile('/home/rkulkarni/v7-scripts/testv7_tlp_config.json')
        if configs:
            run(configs, scrapeURLId)
    except Exception as e:
        log.debug("Exception occurred during execution")
        log.debug("%s", traceback.format_exc())
        traceback.print_exc()
        log.debug("%s", e)
