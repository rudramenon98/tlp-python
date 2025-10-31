## Library import
import io
import json
import logging
import os
import random
import time
import traceback
from datetime import datetime
from multiprocessing import current_process

import pandas as pd
import requests
from common_tools.log_config import configure_logging_from_argv
from database.document_service import (
    cancel_documents,
    find_document_by_url,
    find_documents_not_scraped_on_date,
    get_scrape_script_by_scraperUrlId,
    insert_documents_bulk2,
    update_documents,
    update_documents2,
)
from database.entity.Document import StaticPublicDocument
from database.entity.ScrapScript import ScrapScript
from database.entity.ScriptsProperty import ScriptsConfig, parseCredentialFile
from database.scrape_url_service import (
    scrape_url_append_log,
    update_scrape_url_set_log_value,
)
from PyPDF2 import PdfReader
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
"""
# Console (stdout) handler
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
"""

from database.utils.MySQLFactory import MySQLDriver
from database.utils.util import get_dir_safe
from database.utils.WebDriverFactory import WebDriverFactory

# for logging
logList = []
DateToday = datetime.today()


def download_file_low(URL, path):
    hdr = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
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


def download_file(config: ScriptsConfig, document: StaticPublicDocument, sequence):
    try:
        # mysql_driver = MySQLDriver(cred=config.main_databaseConfig.__dict__)
        file_name = document.title[:50].replace("/", " ") + ".pdf"
        # dir_name = config.rootDataDir + config.downloadDirInRootDir
        dir_name = config.downloadDir
        path = get_dir_safe(dir_name) + "/" + file_name
        log.debug(
            "%s -> Downloading file %s url:%s on location:%s",
            current_process(),
            sequence,
            document.url,
            path,
        )
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
        log.debug("File download failed: %s", document.title + ".pdf")
        log.debug("%s", ex)
        traceback.print_exc()
        return None, str(traceback.format_exc())


def redownload_file(config: ScriptsConfig, document: StaticPublicDocument):
    try:
        # mysql_driver = MySQLDriver(cred=config.main_databaseConfig.__dict__)
        file_name = document.title[:50].replace("/", " ") + ".pdf"
        # dir_name = config.rootDataDir + config.downloadDirInRootDir
        dir_name = config.downloadDir
        path = get_dir_safe(dir_name) + "/" + file_name
        log.debug(
            "%s -> Re-downloading file URL: %s to location: %s",
            current_process(),
            document.url,
            path,
        )
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
        log.debug("File download failed: %s", document.title + ".pdf")
        log.debug("%s", ex)
        traceback.print_exc()
        return None, str(traceback.format_exc())


def get_main_page(main_URL, docker_url):
    driver = WebDriverFactory.getWebDriverInstance(
        browser="docker", docker_url=docker_url
    )

    # wait for the page to fully load
    wait = WebDriverWait(driver, 15)

    # navigate to the main_URL
    driver.get(main_URL)
    wait = WebDriverWait(driver, 20)
    time.sleep(20)

    s = wait.until(
        EC.visibility_of_all_elements_located(
            (By.XPATH, '//*[@id="DataTables_Table_0_paginate"]/ul')
        )
    )
    all_in_one_page = driver.find_element(
        By.XPATH, '//*[@id="DataTables_Table_0_length"]/label/select/option[4]'
    ).click()

    ##Finding number of pages
    all_list_elements = driver.find_elements(
        By.XPATH, '//*[@id="DataTables_Table_0_paginate"]/ul/li/a'
    )
    try:
        number_of_pages = int(all_list_elements[-2].text)
    except ValueError:
        log.debug("L96: Cannot extract number of pages = > %s", len(all_list_elements))
        all_list_elements = s[0].text.split()
        try:
            number_of_pages = int(all_list_elements[-2].text)
        except ValueError:
            log.debug("L101: Cannot extract number of pages = > %s", s[0].text)
            # all_in_one_page = driver.find_element(By.XPATH, '//*[@id="DataTables_Table_0_length"]/label/select/option[5]').click()
            number_of_pages = 0

    log.debug("Number of pages found: %s", str(number_of_pages))

    return driver, number_of_pages


def construct_pdf_filename(doc_url, doc_title):
    # print("doc_url",doc_url,doc_url.split('/'))
    doc_n = doc_url.split("/")[-2]
    doc_filename = doc_n + "-" + doc_title[:49].replace(" ", "_")
    return doc_filename


def check_links(x, docker_url):
    # driver7  =  WebDriverFactory.getWebDriverInstance(browser='chrome')
    driver7 = WebDriverFactory.getWebDriverInstance(
        browser="docker", docker_url=docker_url
    )
    driver7.get(x)
    all_links = []
    new_links = [
        it.get_attribute("href") for it in driver7.find_elements_by_tag_name("a")
    ]
    for j in new_links:
        try:

            if "download" in j:
                download_link = j
                print(download_link)
                all_links.append(download_link)
        except:
            pass
    return all_links


def page_extraction(driver, mysql_driver, scrapeURLId, docker_url):
    global logList

    dataset = {
        "Number": [],
        "title": [],
        "description": [],
        "pdf_file_name": [],
        "pdf_file_url": [],
        "filetype": [],
        "status": [],
        "active_date": [],
        "Download": [],
    }
    table_trs = driver.find_elements_by_xpath('//*[@id="DataTables_Table_0"]/tbody/tr')
    # table_trs = wait.until(EC.visibility_of_all_elements_located((By.XPATH,'//*[@id="DataTables_Table_0"]/tbody/tr/td')))
    # for table in wait.until(EC.visibility_of_all_elements_located((By.XPATH,'//*[@id="DataTables_Table_0"]/tbody/tr/td'))):
    download = True
    for i in table_trs:
        #         try:
        try:
            td = [item.text for item in i.find_elements_by_xpath(".//td")]
            td1 = [item for item in i.find_elements_by_xpath(".//td")]
            link = [
                item.get_attribute("href")
                for item in i.find_elements_by_xpath(
                    '//*[@id="DataTables_Table_0"]/tbody/tr/td[2]/a'
                )
            ]

            link2 = [
                it.get_attribute("href") for it in i.find_elements_by_tag_name("a")
            ]
            download_link = ""
            for j in link2:
                if "download" in j:
                    download_link = j
                    doc_number = j.split("/")[-2]
                    break

            if len(download_link) == 0:
                # logText = 'Skipped Title: ' + str(td[0]) + ' PDF URL: NOT FOUND on Main Page' + '  on ' + str(datetime.today())
                # logList.append(logText)
                # scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                # print(logText)

                url_next_page = (
                    td1[0].find_elements_by_xpath(".//a")[0].get_attribute("href")
                )
                links = check_links(url_next_page, docker_url)
                for k in links:

                    dataset["Number"].append(k.split("/")[-2])

                    dataset["title"].append(td[0][:100])

                    dataset["description"].append(td[3])

                    dataset["pdf_file_name"].append("")

                    dataset["pdf_file_url"].append(k)

                    dataset["filetype"].append(3)  # Guidance Documents

                    dataset["status"].append(td[4])

                    dataset["active_date"].append(
                        datetime.strptime(td[2], "%m/%d/%Y").date()
                    )

                    dataset["Download"].append(download)

                continue

            dataset["Number"].append(doc_number)

            dataset["title"].append(td[0][:100])

            dataset["description"].append(td[3])

            dataset["pdf_file_name"].append("")

            dataset["pdf_file_url"].append(download_link)

            dataset["filetype"].append(3)  # Guidance Documents

            dataset["status"].append(td[4])

            dataset["active_date"].append(datetime.strptime(td[2], "%m/%d/%Y").date())

            dataset["Download"].append(download)

            # result.append(row)

            logText = (
                "Scraped Title: "
                + str(td[0])
                + " PDF URL: "
                + str(download_link)
                + "  on "
                + str(datetime.today())
            )
            logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            print(logText)
        except:
            print("Row is not Parsed")
            logText = "Row is not Parsed " + doc_number + " href =  " + link
            logText += "\n " + traceback.format_exc()
            logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            print(logText)

    # print(dataset)
    df = pd.DataFrame.from_dict(dataset, orient="index")
    df = df.transpose()
    # df = pd.DataFrame(dataset)
    print(df)
    # driver4.quit()
    return df


def process_page(config: ScriptsConfig, dfs, mysql_driver, scrapeURLId):

    global logList, DateToday

    print("Processing website page")
    time.sleep(60 + 105 * random.random())
    cnt = 0
    scrapeDF = dfs
    scrapeDF["ID"] = [i for i in range(len(scrapeDF))]
    scrapeDF["pdf_file_name"] = scrapeDF.apply(
        lambda x: construct_pdf_filename(x["pdf_file_url"], x["title"]), axis=1
    )
    scrapeDF["ScrapingLog"] = ""
    scrapeDF[["description", "Number"]] = scrapeDF[["description", "Number"]].fillna("")

    mysql_driver2 = MySQLDriver(cred=config.databaseConfig.__dict__)
    scrapeScript: ScrapScript = get_scrape_script_by_scraperUrlId(
        mysql_driver2, scrapeURLId
    )

    # check for new documents & unchanged documents
    update_list, download_list, redownload_list = check_for_new_documents(
        mysql_driver2,
        scrapeDF,
        scrapeURLId,
        scrapeScript,
        download_dir_name=config.downloadDir,
    )

    # update the unchanged documents in DB (just set their lastScrapeDate = DateToday)
    if update_list and len(update_list) > 0:
        update_documents(mysql_driver2, update_list, doc_class=StaticPublicDocument)

    redownloaded_documents = []
    recnt = 0
    for rdoc in redownload_list:
        print("DOC:", str(rdoc.number))
        recnt += 1
        if recnt % 5:
            time.sleep(60 + 105 * random.random())
        doc, msgText = redownload_file(config, rdoc)
        if doc is not None:
            redownloaded_documents.append(doc)
        else:
            logList.append(msgText)
            scrape_url_append_log(mysql_driver2, scrapeURLId, msgText)

    if len(redownloaded_documents) > 0:
        update_documents2(
            mysql_driver2, redownloaded_documents, doc_class=StaticPublicDocument
        )

    print("final download  list size:")
    print(download_list)
    print(len(download_list))

    bulk_document = []
    logText = f"File downloading started at {datetime.today()}"
    logList.append(logText)
    scrape_url_append_log(mysql_driver2, scrapeURLId, logText)

    t1 = time.time()
    print("starting download files")
    # cnt = 0
    for doc in download_list:
        print("DOC:", str(doc.number))
        cnt += 1
        if cnt % 5:
            time.sleep(60 + 105 * random.random())

        docs, msgText = download_file(config, doc, cnt)
        if docs is not None:
            bulk_document.append(docs)
        else:
            logList.append(msgText)
            scrape_url_append_log(mysql_driver2, scrapeURLId, msgText)

    logText = f"File downloading completed at {datetime.today()} time taken: {str((time.time() - t1))}"
    logList.append(logText)
    scrape_url_append_log(mysql_driver2, scrapeURLId, logText)
    print(logText)

    docIDsList = None
    try:
        if bulk_document is not None and len(bulk_document) > 0:
            docIDsList = insert_documents_bulk2(mysql_driver2, bulk_document)

        if docIDsList is not None and len(docIDsList) > 0:
            print(f"Newly added docIDs: {docIDsList}")
            logText = f"Newly added docIDs: {docIDsList}"
            logList.append(logText)
            scrape_url_append_log(mysql_driver2, scrapeURLId, logText)
    except:
        logText = traceback.format_exc()
        logList.append(logText)
        scrape_url_append_log(mysql_driver2, scrapeURLId, logText)
        print(logText)

    # handle cancelled documents
    cancel_list = check_for_cancelled_documents(
        mysql_driver, DateToday, scrapeURLId, scrapeScript
    )

    if cancel_list is not None:
        logText = f"Number of cancelled documents since last scrape: {len(cancel_list)}"
        logList.append(logText)
        scrape_url_append_log(mysql_driver2, scrapeURLId, logText)
        print(logText)
    else:
        logText = f"Number of cancelled documents since last scrape: 0"
        logList.append(logText)
        scrape_url_append_log(mysql_driver2, scrapeURLId, logText)
        print(logText)

    # Mark the newly cancelled documents in the documents table
    if cancel_list and len(cancel_list) > 0:
        cancel_documents(mysql_driver2, cancel_list, doc_class=StaticPublicDocument)

    mysql_driver2.close()
    print("Processed website page")


def scrape_fda_guidance_documentsOLD(
    config: ScriptsConfig,
    driver,
    number_of_pages,
    mysql_driver,
    scrapeURLId,
    docker_url,
):
    # dfs = []
    try:

        for _ in range(number_of_pages):

            print("Scraping page: " + str(_))
            df = page_extraction(driver, mysql_driver, scrapeURLId, docker_url)

            # print(df)
            process_page(config, df, mysql_driver, scrapeURLId)

            next_page = driver.find_element(
                By.XPATH, '//*[@id="DataTables_Table_0_next"]/a'
            ).click()
            time.sleep(30)
    except Exception:
        logText = traceback.format_exc()
        logList.append(logText)
        print(logText)
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    # return dfs


def scrape_fda_guidance_documents(
    config: ScriptsConfig,
    driver,
    number_of_pages,
    mysql_driver,
    scrapeURLId,
    docker_url,
):
    # dfs = []
    all_dfs = []
    try:
        # Step 1: Scrape and collect all pages
        for page_num in range(number_of_pages):
            print(f"Scraping page: {page_num + 1}")

            df = page_extraction(driver, mysql_driver, scrapeURLId, docker_url)
            all_dfs.append(df)

            #            if page_num < number_of_pages - 1:
            next_page = driver.find_element(
                By.XPATH, '//*[@id="DataTables_Table_0_next"]/a'
            ).click()
            time.sleep(30)

        # Step 2: Process each page's DataFrame individually
        for df in all_dfs:
            process_page(config, df, mysql_driver, scrapeURLId)

    except Exception:
        logText = traceback.format_exc()
        logList.append(logText)
        print(logText)
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    # return dfs


def check_file_on_disk_is_pdfOLD(file_name: str, download_dir: str):
    try:
        with open(download_dir + file_name, "rb") as f:
            data = f.read(50)
            html_file = str(data).startswith("<!DOCTYPE html>")
            return not html_file
        return True
    except Exception:
        traceback.print_exc()
        return True


def check_file_on_disk_is_pdf(file_name: str, download_dir: str):
    try:
        PdfReader(download_dir + file_name)
        return True
    except:
        traceback.print_exc()
        return False


def check_for_new_documents(
    mysql_driver,
    scrapeDF,
    scrapeURLId,
    scrapeScript: ScrapScript,
    download_dir_name: str,
):
    global logList
    download_list = []
    update_list = []
    skip_list = []
    redownload_list = []

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
                    print(logText)
                    continue

            docInDB = find_document_by_url(
                mysql_driver, file_url, doc_class=StaticPublicDocument
            )
        except Exception:
            logText = f"Failed for : {file_url} \n"
            logText += traceback.format_exc()
            logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            print(logText)
            continue

        # scrapeScript: ScrapScript = get_scrap_script_by_file_name(mysql_driver, os.path.basename(__file__))

        if not docInDB:
            try:
                document = StaticPublicDocument(
                    number=row["Number"],
                    title=row["title"],
                    description=row["description"][:1024],
                    url=row["pdf_file_url"],
                    # documentType=row['filetype'],
                    documentType=scrapeScript.documentTypeID,
                    documentStatus=1,  # Active
                    activeDate=row["active_date"],
                    inactiveDate=None,
                    sourceFileName=None,
                    pdfFileName=None,
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
            except Exception:
                logText = (
                    f"New StaticPublicDocument row creation failure for : {file_url} \n"
                )
                logText += traceback.format_exc()
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                print(logText)

        else:
            isPdfFile = check_file_on_disk_is_pdf(
                docInDB.pdfFileName, download_dir_name
            )
            if isPdfFile:
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
            else:
                docInDB.lastScrapeDate = datetime.today().date()
                logText = (
                    "File "
                    + row["title"]
                    + " with Guidance Doc at "
                    + row["pdf_file_url"]
                    + " scraped and PDF file re-downloaded on  "
                    + str(datetime.today())
                )
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                print(logText)

                redownload_list.append(docInDB)

    return update_list, download_list, redownload_list


def check_if_file_exists(file_url):
    retValue = False
    if file_url and len(file_url) > 0:
        r = requests.get(file_url, stream=True)

        if r.status_code == 200:
            retValue = True
        else:
            retValue = False
    else:
        retValue = False
    return retValue


def check_if_file_exists3(link, mysql_driver, scrapeURLId):
    # link = 'https://www.govinfo.gov/content/pkg/CFR-'+str(year)+'-title'+str(title)+'-vol'+str(vol)+'/pdf/CFR-'+str(year)+'-title'+str(title)+'-vol'+str(vol)+'.pdf'
    # print(link)
    hdr = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36",
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
                + " Downloaded "
                + str(size)
                + "out of "
                + str(total)
            )
            logText = (
                "Warning "
                + str(link)
                + " Downloaded "
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


def check_for_cancelled_documents(
    mysql_driver, current_date, scrapeURLId, scrapeScript: ScrapScript
):
    global logList

    cancel_list = []

    # get the documents in documents table which were not scraped today
    old_documents_list = find_documents_not_scraped_on_date(
        mysql_driver, current_date, doc_class=StaticPublicDocument
    )

    if not old_documents_list or len(old_documents_list) == 0:
        return None

    for old_doc in old_documents_list:
        old_doc.number
        old_url = old_doc.url

        if old_doc.documentType != scrapeScript.documentTypeID:
            # skip the non Guidance Documents
            continue

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


def save_log_data_to_db(log_list, scrapeURLId, mysql_driver):
    # Passing ScrapeURL ID hard-coded as it is already declared and won't be changed
    update_scrape_url_set_log_value(mysql_driver, scrapeURLId, json.dumps(log_list))


def run(config: ScriptsConfig, scrapeURLId):
    global logList
    datetime.today().date()

    main_URL = "https://www.fda.gov/medical-devices/device-advice-comprehensive-regulatory-assistance/guidance-documents-medical-devices-and-radiation-emitting-products"

    # get the number of pages
    driver, number_of_pages = get_main_page(main_URL, config.SeleniumDocker)
    # number_of_pages = 10
    mysql_driver = MySQLDriver(cred=config.databaseConfig.__dict__)

    logText = f"Scrape for FDA Guidance Documents started at {datetime.today()} URL: {main_URL}"
    logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    print(logText)

    logText = f"Website data captured at {datetime.today()}"
    logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)

    # scrape the FDA website: page by page
    scrape_fda_guidance_documents(
        config,
        driver,
        number_of_pages,
        mysql_driver,
        scrapeURLId,
        config.SeleniumDocker,
    )

    # Mark the newly cancelled documents in the documents table
    # if cancel_list and len(cancel_list) > 0:
    #    cancel_documents(mysql_driver, cancel_list)

    # scrapeDF.to_csv('FDA_GD_web_Scrapping.csv', index=False)
    logText = f"Scraping URL: {main_URL} DONE"
    logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    print(logText)

    # save_log_data_to_db(logList, mysql_driver)
    print("Scaped URL:" + main_URL)


def parse_remaining_args(cleaned_args):
    repo_id = None
    values = []

    i = 0
    while i < len(cleaned_args):
        if cleaned_args[i] == "--repo_id":
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
        props = None

        # configure the logging level
        remaining_args = configure_logging_from_argv(default_level="INFO")
        repo_id, docIdsList = parse_remaining_args(remaining_args)

        if len(docIdsList) > 0:
            scrapeURLId = docIdsList[0]
        else:
            scrapeURLId = 2

        configs = parseCredentialFile("/app/tlp_config.json")

        if configs:
            run(configs, scrapeURLId)

    except Exception as e:
        print("The EXCEPTION >>>>>>>>>>>>>> ")
        print(traceback.format_exc())
        traceback.print_exc()
        print(e)
