import io
import logging
import os
import time
import traceback
from datetime import datetime

import pandas as pd
import requests
from common_tools.log_config import configure_logging_from_argv
from database.document_service import (
    find_document_by_url,
    find_documents_not_scraped_on_date,
    get_scrape_script_by_scraperUrlId,
    insert_document,
)
from database.entity.Document import Document
from database.entity.ScrapScript import ScrapScript
from database.entity.ScriptsProperty import ScriptsConfig, parseCredentialFile
from database.scrape_url_service import scrape_url_append_log
from database.utils.MySQLFactory import MySQLDriver
from database.utils.util import get_dir_safe
from database.utils.WebDriverFactory import WebDriverFactory
from dateutil.parser import parse
from PyPDF2 import PdfReader
from selenium.webdriver.common.by import By
from tqdm import tqdm

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

"""
# Console (stdout) handler
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
"""

logList = []
DateToday = datetime.today()
year = DateToday.year


# def download_file(config: ScriptsConfig, document: Document, sequence):
#     try:
#         #mysql_driver = MySQLDriver(cred=config.main_databaseConfig.__dict__)
#         file_name = document.title[:50].replace('/',' ') + ".pdf"
#         #dir = config.rootDataDir + config.downloadDirInRootDir
#         dir = config.downloadDir
#         path = get_dir_safe(dir) + "/" + file_name
#         print(f"{current_process()} -> Downloading file {sequence} url:{document.url} on location:{path}")
#         time.sleep(10)
#         response = urllib.request.urlopen(document.url)
#         file = open(path, 'wb')
#         file.write(response.read())
#         file.close()
#         document.pdfFileName = os.path.relpath(path, config.downloadDir)
#         document.fileSize = os.path.getsize(path)
#         document.parsed = False
#         #insert_document(mysql_driver, document)
#         #mysql_driver.close()
#         return document, 'Success'

#     except Exception as ex:
#         print('File is not Downloaded:',document.title + ".pdf")
#         log.debug("%s", ex)
#         traceback.print_exc()
#         return None, str(traceback.format_exc())


def download_file(URL, path):
    hdr = {
        #'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36',
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9,gu;q=0.8,hi;q=0.7",
        "Connection": "keep-alive",
    }
    chunk_size = (1024 * 1024) * 1  # 1MB
    try:
        response = requests.get(
            URL,
            headers=hdr,
            verify="/app/scripts/venv/lib/python3.9/site-packages/certifi/cacert.pem",
            timeout=60,
        )
        # response = requests.get(URL, headers=hdr,verify = '/etc/ssl/cert.pem', timeout=60)
        total = int(response.headers.get("content-length", 0))
        filename = URL.split("/")[-1]
        path = path + "/" + filename
        with open(path, "wb") as file, tqdm(
            desc=path,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)
        return True, "Success"
    except Exception as ex:
        log.debug("File is not Downloaded: %s", URL)
        log.debug("%s", ex)
        traceback.print_exc()
        return False, str(traceback.format_exc())


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def initial_page_scrape(driver, url):
    driver.get(url)
    # wait = WebDriverWait(driver,10)
    time.sleep(10)
    all_list_elements = driver.find_elements(
        By.XPATH,
        '//*[@id="block-mainpagecontent"]/article/div/div[2]/div[2]/div/div[3]/div/div/div/div/div/div/ul/li',
    )
    # print(all_list_elements)
    # driver.close()
    return all_list_elements


def file_doc_url_and_name(url):
    # driver2 = webdriver.Chrome('/home/dshah/nlp/chrome_driver/chromedriver',options=chrome_options)
    driver2 = WebDriverFactory.getWebDriverInstance(browser="docker")
    driver2.get(url)
    # wait = WebDriverWait(driver2,10)
    # time.sleep(10)
    pdfs = []
    pdf_names = []

    docxs = []
    docx_names = []
    x = driver2.find_elements_by_tag_name("a")
    for i in x:
        # print(i.get_attribute.__dict__)
        try:
            if i.get_attribute("href").endswith("pdf"):
                pdf_url = i.get_attribute("href")
                pdfs.append(pdf_url)
                pdf_name = pdf_url.split("/")[-1]
                pdf_names.append(pdf_name)
            elif i.get_attribute("href").endswith("docx"):
                docx_url = i.get_attribute("href")
                docxs.append(docx_url)
                docx_name = docx_url.split("/")[-1]
                docx_names.append(docx_name)

        #                file_path = '/home/dshah/Inspird-2023-dev/Web_Scrapping/AUS_MDR_DOCS/'+str(doc_name)
        # download_file(doc_url,file_path)
        except Exception as ex:
            log.debug("%s", ex)
            traceback.print_exc()
    log.debug("%s", docs)
    return pdfs, pdf_names, docxs, docx_names


def regulation_Scrapping(links):
    links = [
        "https://www.tga.gov.au/sites/default/files/devices-argmd-01.pdf",
        "https://www.tga.gov.au/sites/default/files/devices-argmd-01.docx",
    ]
    data = {
        "title": [],
        "discription": [],
        "source_file_name": [],
        "source_file_url": [],
        "pdf_file_name": [],
        "pdf_file_url": [],
        "filetype": [],
        "filetype": [],
        "status": [],
        "active_date": [],
        "Download": [],
        "ID": [],
        "Number": [],
    }
    title = "Australia Regulations"
    active_date = "1 January " + str(year)
    discription = ""
    data["title"].append(title)
    data["discription"].append(discription)
    data["source_file_name"].append("devices-argmd-01.docx")
    data["source_file_url"].append(links[1])
    data["pdf_file_name"].append("devices-argmd-01.pdf")
    data["pdf_file_url"].append(links[0])
    data["filetype"].append("pdf")
    data["status"].append("active")
    data["active_date"].append(active_date)
    data["Download"].append(True)
    data["ID"].append(1)
    data["Number"].append("")
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.transpose()
    return df


def check_for_new_documents(
    mysql_driver, scrapeDF, scrapeURLId, scrapeScript: ScrapScript
):
    # global logList
    download_list = []
    update_list = []
    skip_list = []

    for idx, row in scrapeDF.iterrows():
        file_url = row["pdf_file_url"]

        try:
            #            if row['Number'] == 0:
            #                continue

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
                    # logList.append(logText)
                    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                    log.debug("%s", logText)
                    continue

            docInDB = find_document_by_url(mysql_driver, file_url)

        except Exception:
            logText = f"Failed for : {file_url} \n"
            logText += traceback.format_exc()
            # logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            print(logText)
            continue

        # scrapeScript: ScrapScript = get_scrap_script_by_file_name(mysql_driver, os.path.basename(__file__))

        if not docInDB:
            print(f"{file_url} not found in DB")
            try:
                document = Document(
                    number=row["Number"],
                    title=row["title"],
                    description=str(row["discription"][:1024]),
                    url=str(row["pdf_file_url"]),
                    # documentType=row['filetype'],
                    documentType=scrapeScript.documentTypeID,
                    documentStatus=1,  # Active
                    activeDate=datetime.strptime(row["active_date"], "%d %B %Y"),
                    inactiveDate=None,
                    sourceFileName=str(row["source_file_name"]),
                    pdfFileName=str(row["pdf_file_name"]),
                    parsed=False,
                    embedded=False,
                    # parsingScriptID=scrapeScript.defaultParsingScriptID,
                    parsingScriptID=scrapeScript.defaultParsingScriptID,
                    createdDate=datetime.today().date(),
                    modifiedDate=datetime.today().date(),
                    # additionalInfo=row['pdf_file_name'],
                    fileSize=0,
                    parsingLog="notParsed",
                    embeddingLog="notEmbedded",
                    noOfParagraphs=0,
                    lastScrapeDate=datetime.today().date(),
                    # scrapingLog = 'scraped successfully',
                    sourceProject=0,
                )

                download_list.append(document)
                print(type(document))
                #                insert_document(mysql_driver, document)

                logText = (
                    str(row["pdf_file_name"])
                    + " with filetype = "
                    + str(row["filetype"])
                    + " at "
                    + str(row["pdf_file_url"][0])
                    + " downloaded on "
                    + str(datetime.today().date())
                )
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            except Exception:
                logText = f"New Document row creation failed for : {file_url} \n"
                logText += traceback.format_exc()
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                print(logText)

        else:
            print(f"{file_url} found in DB")
            docInDB.lastScrapeDate = datetime.today().date()
            logText = (
                "File "
                + row["title"]
                + " with Guidance Doc at "
                + row["pdf_file_url"]
                + " scraped and not changed on "
                + str(datetime.today())
            )
            logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            print(logText)
            update_list.append(docInDB)

    return update_list, download_list


def check_if_file_exists3(link, mysql_driver, scrapeURLId):
    # link = 'https://www.govinfo.gov/content/pkg/CFR-'+str(year)+'-title'+str(title)+'-vol'+str(vol)+'/pdf/CFR-'+str(year)+'-title'+str(title)+'-vol'+str(vol)+'.pdf'
    # print(link)
    hdr = {
        #'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36',
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9,gu;q=0.8,hi;q=0.7",
        "Connection": "keep-alive",
    }

    # page = requests.get(str(link), headers=hdr)

    chunk_size = (1024 * 1024) * 1  # 1MB

    response = requests.get(str(link), headers=hdr, stream=True)
    total = int(response.headers.get("content-length", 0))

    with io.BytesIO() as stream:
        size = 0
        for data in response.iter_content(chunk_size=chunk_size):
            size += stream.write(data)

        if total != size:
            print(
                "Warning "
                + str(link)
                + " Downladed "
                + str(size)
                + "out of "
                + str(total)
            )
            logText = (
                "Warning "
                + str(link)
                + " Downladed "
                + str(size)
                + "out of "
                + str(total)
            )
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
        try:
            PdfReader(stream)
            # print(pdf)
            return True
        except:
            # print('Year',year,'Title',title,'Vol',vol,'is not found')
            return False

    return False


def check_for_cancelled_documents(mysql_driver, current_date, scrapeURLId):
    global logList

    cancel_list = []

    # get the documents in documents table which were not scraped today
    old_documents_list = find_documents_not_scraped_on_date(mysql_driver, current_date)

    if not old_documents_list or len(old_documents_list) == 0:
        return None

    for old_doc in old_documents_list:
        old_doc.number
        old_url = old_doc.url

        # check if url exists, i.e. guidance document is downloadable from FDA website
        doc_exists = check_if_file_exists3(old_url, mysql_driver, scrapeURLId)

        if doc_exists == False:
            # Mark as Inactive
            old_doc.documentStatus = 2
            old_doc.inactiveDate = datetime.today().date()
            old_doc.modifiedDate = datetime.today().date()

            # write to log
            logText = (
                "File "
                + str(old_doc.pdfFileName)
                + " with "
                + str(old_doc.documentType)
                + " changed to withdrawn/cancelled on "
                + str(datetime.today())
            )
            logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            print(logText)

            # append to cancel list
            cancel_list.append(old_doc)

    return cancel_list


def run(config: ScriptsConfig, scrapeURLId):
    # global logList

    datetime.today().date()
    mysql_driver = MySQLDriver(cred=config.databaseConfig.__dict__)
    scrapeScript: ScrapScript = get_scrape_script_by_scraperUrlId(
        mysql_driver, scrapeURLId
    )

    links = [
        "https://www.tga.gov.au/sites/default/files/devices-argmd-01.pdf",
        "https://www.tga.gov.au/sites/default/files/devices-argmd-01.docx",
    ]

    main_URL = " https://www.tga.gov.au/resources/publication/publications/australian-regulatory-guidelines-medical-devices-argmd-v11-archived-version "

    logText = f"ScrapeID: {scrapeURLId}: Scrape for Australia MDR Documents started at {datetime.today()} URL: {main_URL}"

    # logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    print(logText)

    # get all the documents and their details
    scrapeDF = regulation_Scrapping(links)

    logText = f"Website data captured at {datetime.today()}"
    # logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    print(logText)
    # download_list_link  = [item for sublist in list_of_urls for item in sublist if item]
    update_list, download_list = check_for_new_documents(
        mysql_driver, scrapeDF, scrapeURLId, scrapeScript
    )

    # #update the unchanged documents in DB (just set their lastScrapeDate = DateToday)
    # if update_list and len(update_list) > 0:
    #     update_documents(mysql_driver, update_list)

    print("final download  list size:")
    print(download_list)
    print(len(download_list))

    logText = f"File downloading started at {datetime.today()}"
    # logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)

    t1 = time.time()
    print("starting download files")
    cnt = 0
    dir = config.downloadDir
    path = get_dir_safe(dir)
    for doc in links:
        # print("DOC:", str(doc.number))
        cnt += 1
        download_file(doc, path)
        # bulk_document.append(docs)

    pdfFilePath = path + "/" + download_list[0].pdfFileName
    download_list[0].pdfFileName = os.path.relpath(pdfFilePath, config.downloadDir)
    srcFilePath = path + "/" + download_list[0].sourceFileName
    download_list[0].sourceFileName = os.path.relpath(srcFilePath, config.downloadDir)
    download_list[0].fileSize = os.path.getsize(pdfFilePath)
    insert_document(mysql_driver, download_list[0])
    logText = f"File downloading completed at {datetime.today()} time taken: {str((time.time() - t1))}"
    # logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    print(logText)


#     update_list, download_list = check_for_new_documents(mysql_driver, scrapeDF, scrapeURLId, scrapeScript)

#     #update the unchanged documents in DB (just set their lastScrapeDate = DateToday)
#     if update_list and len(update_list) > 0:
#         update_documents(mysql_driver, update_list)

#     print("final download  list size:")
#     print(download_list)
#     print(len(download_list))

#     bulk_document = []
#     logText = f'File downloading started at {datetime.today()}'
#     #logList.append(logText)
#     scrape_url_append_log(mysql_driver, scrapeURLId, logText)

#     t1 = time.time()
#     print("starting download files")
#     cnt = 0
#     for doc in download_list:
#         print("DOC:", str(doc.number))
#         cnt += 1
#         docs, msgText = download_file(config,doc,cnt)
#         if docs is not None:
#             bulk_document.append(docs)
#         else:
#             logList.append(msgText)
#             scrape_url_append_log(mysql_driver, scrapeURLId, msgText)
# #        bulk_document.append(docs)

#     logText = f'File downloading completed at {datetime.today()} time taken: {str((time.time() - t1))}'
#     #logList.append(logText)
#     scrape_url_append_log(mysql_driver, scrapeURLId, logText)
#     print(logText)

#     docIDsList = None
#     try:
#         if bulk_document is not None and len(bulk_document) > 0:
#             docIDsList = insert_documents_bulk2(mysql_driver, bulk_document)

#         if docIDsList is not None and len(docIDsList) > 0:
#             logText = f'Newly added docIDs: {docIDsList}'
#             #logList.append(logText)
#             scrape_url_append_log(mysql_driver, scrapeURLId, logText)
#             print(logText)
#     except:
#         logText = traceback.format_exc()
#         #logList.append(logText)
#         scrape_url_append_log(mysql_driver, scrapeURLId, logText)
#         print(logText)

#     # handle cancelled documents
#     cancel_list = check_for_cancelled_documents(mysql_driver, DateToday, scrapeURLId)

#     logText = f'Number of cancelled documents since last scrape: {len(cancel_list)}'
#     logList.append(logText)
#     scrape_url_append_log(mysql_driver, scrapeURLId, logText)
#     print(logText)

#     # Mark the newly cancelled documents in the documents table
#     if cancel_list and len(cancel_list) > 0:
#         cancel_documents(mysql_driver, cancel_list)


#     logText = f'Scraping URL: {main_URL} DONE'
#     #logList.append(logText)
#     scrape_url_append_log(mysql_driver, scrapeURLId, logText)
#     print(logText)

#     #save_log_data_to_db(logList, mysql_driver)
#     print('Scaped URL:' + main_URL)


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
            scrapeURLId = 8

        configs = parseCredentialFile("/app/tlp_config.json")

        if configs:
            run(configs, scrapeURLId)
    except Exception as e:
        print("The EXCEPTION >>>>>>>>>>>>>> ")
        print(traceback.format_exc())
        traceback.print_exc()
        print(e)
