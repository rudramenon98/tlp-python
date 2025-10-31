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
    get_parsing_script_by_document_type_name,
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


def download_file(URL, path):
    hdr = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        #        'Accept-Encoding': 'gzip, deflate, br',
        "Accept-Language": "en-US,en;q=0.9,gu;q=0.8,hi;q=0.7",
        "Connection": "keep-alive",
    }
    chunk_size = (1024 * 1024) * 1  # 1MB
    try:
        # response = requests.get(URL, headers=hdr,verify = '/app/scripts/venv/lib/python3.9/site-packages/certifi/cacert.pem')
        response = requests.get(URL, headers=hdr, verify="/etc/ssl/cert.pem")
        total = int(response.headers.get("content-length", 0))
        URL.split("/")[-1]
        # path = path+'/'+filename
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


def file_doc_url_and_name(url, docker_url):
    # driver2 = webdriver.Chrome('/home/dshah/nlp/chrome_driver/chromedriver',options=chrome_options)
    driver2 = WebDriverFactory.getWebDriverInstance(
        browser="docker", docker_url=docker_url
    )
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
    log.debug(
        "Found URLs - PDFs: %s, PDF names: %s, DOCX URLs: %s, DOCX names: %s",
        pdfs,
        pdf_names,
        docxs,
        docx_names,
    )
    # print(docs)
    return pdfs, pdf_names, docxs, docx_names


def MDR_Scrapping(url, seleniumDocker: str):
    data = {
        "Number": [],
        "title": [],
        "discription": [],
        "source_file_name": [],
        "source_file_url": [],
        "pdf_file_name": [],
        "pdf_file_url": [],
        "filetype": [],
        "status": [],
        "active_date": [],
        "Download": [],
        "ID": [],
    }

    cn = 0
    URL = url

    for page_no in range(10):
        try:
            log.debug("Scraping page number: %s", page_no)

            # url = 'https://www.tga.gov.au/resources/resource?f%5B0%5D=topics%3A225&page='+str(page_no)
            driver = WebDriverFactory.getWebDriverInstance(
                browser="docker", docker_url=seleniumDocker
            )
            if page_no > 0:
                url = URL + "&page=" + str(page_no)
            else:
                url = URL
            log.debug("Processing URL: %s", url)
            all_elements = initial_page_scrape(driver, url)

            # print(all_elements)

            for i in all_elements:
                values = list(i.text.split("\n"))
                doc_url = i.find_element_by_tag_name("a").get_attribute("href")
                print(doc_url)
                pdf_url, pdf_name, docx_url, docx_name = file_doc_url_and_name(
                    doc_url, docker_url=seleniumDocker
                )
                if len(values) == 4:
                    title = values[0]
                    active_date = values[1]
                    discription = values[-1]
                    data["title"].append(title)
                    data["discription"].append(discription)
                    data["source_file_name"].append(docx_name)
                    data["source_file_url"].append(docx_url)
                    data["pdf_file_name"].append(pdf_name)
                    data["pdf_file_url"].append(pdf_url)
                    data["filetype"].append("pdf")
                    data["status"].append("active")
                    data["active_date"].append(active_date)
                    data["Download"].append(True)
                    data["ID"].append(cn)
                    data["Number"].append("")
                elif len(values) == 3:
                    print(values, title)
                    title = values[0]
                    if is_date(str(values[1])):
                        active_date = values[1]
                        discription = ""
                    else:
                        active_date = "01 January " + str(year)
                        discription = values[1]

                    data["title"].append(title)
                    data["discription"].append(discription)
                    data["source_file_name"].append(docx_name)
                    data["source_file_url"].append(docx_url)
                    data["pdf_file_name"].append(pdf_name)
                    data["pdf_file_url"].append(pdf_url)
                    data["filetype"].append("pdf")
                    data["status"].append("active")
                    data["active_date"].append(active_date)
                    data["Download"].append(True)
                    data["ID"].append(cn)
                    data["Number"].append("")
                else:  # len(values) == 2:
                    print(values, title)

                    title = values[0]
                    active_date = "1/1/" + str(year)
                    discription = ""
                    data["title"].append(title)
                    data["discription"].append(discription)
                    data["source_file_name"].append(docx_name)
                    data["source_file_url"].append(docx_url)
                    data["pdf_file_name"].append(pdf_name)
                    data["pdf_file_url"].append(pdf_url)
                    data["filetype"].append("pdf")
                    data["status"].append("active")
                    data["active_date"].append(active_date)
                    data["Download"].append(True)
                    data["ID"].append(cn)
                    data["Number"].append("")
                cn += 1

        except Exception as e:
            print("Done with the Page", e)
            continue

    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.transpose()
    return df


def check_for_new_documents(
    config, mysql_driver, scrapeDF, scrapeURLId, scrapeScript: ScrapScript
):
    # global logList
    download_list = []
    update_list = []
    skip_list = []

    db_name = config.databaseConfig.__dict__["database"]
    sql_stmt = f"SELECT parsingScriptID from {db_name}.parsingscripts where title like '%PDF%';"
    result = get_parsing_script_by_document_type_name(mysql_driver, sql_stmt)

    for r in result:
        PDF_parsingScriptID = r[0]
        break

    for idx, row in scrapeDF.iterrows():
        defaultParsingScriptID = scrapeScript.defaultParsingScriptID

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
                    print(logText)
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
            DOCX_URL = row["source_file_url"]
            if isinstance(DOCX_URL, list):
                if len(DOCX_URL) > 0:
                    DOCX_URL = str(DOCX_URL[0])

                else:
                    DOCX_URL = ""
                    defaultParsingScriptID = PDF_parsingScriptID  # Parsed by PDF Parser as docx is not available

            PDF_URL = str(file_url)

            file_name = DOCX_URL.split("/")[-1]
            pdf_file_name = PDF_URL.split("/")[-1]

            # folder = config.rootDataDir + config.downloadDirInRootDir
            folder = config.downloadDir
            path = get_dir_safe(folder) + "/" + file_name
            pdf_path = get_dir_safe(folder) + "/" + pdf_file_name

            try:
                download_file(DOCX_URL, path)
            except Exception:
                logText = (
                    file_name
                    + " with filetype = "
                    + str(row["filetype"])
                    + " at "
                    + DOCX_URL
                    + " download failed on "
                    + str(datetime.today().date())
                )
                logText += traceback.format_exc()
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                print(logText)
                path = None

            try:
                download_file(PDF_URL, pdf_path)
            except Exception:
                logText = (
                    pdf_file_name
                    + " with filetype = "
                    + str(row["filetype"])
                    + " at "
                    + PDF_URL
                    + " download failed on "
                    + str(datetime.today().date())
                )
                logText += traceback.format_exc()
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                print(logText)
                continue

            try:
                document = Document(
                    number=row["Number"],
                    title=row["title"],
                    description=str(row["discription"][:1024]),
                    # url=str(row['pdf_file_url'][0]),
                    url=str(PDF_URL),
                    # documentType=row['filetype'],
                    documentType=scrapeScript.documentTypeID,
                    documentStatus=1,  # Active
                    activeDate=datetime.strptime(row["active_date"], "%d %B %Y"),
                    inactiveDate=None,
                    sourceFileName=os.path.relpath(path, config.downloadDir),
                    pdfFileName=os.path.relpath(pdf_path, config.downloadDir),
                    parsed=False,
                    embedded=False,
                    # parsingScriptID=scrapeScript.defaultParsingScriptID,
                    parsingScriptID=defaultParsingScriptID,
                    createdDate=datetime.today().date(),
                    modifiedDate=datetime.today().date(),
                    # additionalInfo=row['pdf_file_name'],
                    fileSize=os.path.getsize(pdf_path),
                    parsingLog="notParsed",
                    embeddingLog="notEmbedded",
                    noOfParagraphs=0,
                    lastScrapeDate=datetime.today().date(),
                    # scrapingLog = 'scraped successfully',
                    sourceProject=0,
                )

                download_list.append(document)
                print(type(document))

                insert_document(mysql_driver, document)

                logText = (
                    str(pdf_file_name)
                    + " with filetype = "
                    + str(scrapeScript.documentTypeID)
                    + " at "
                    + str(PDF_URL)
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
                + str(row["pdf_file_url"])
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

    main_URL = "https://www.tga.gov.au/resources/resource?f%5B0%5D=topics%3A225&page="
    main_URL = "https://www.tga.gov.au/resources/guidance?f%5B0%5D=guidance_product_type%3A1305"

    logText = f"ScrapeID: {scrapeURLId}: Scrape for Australia Guidance Documents started at {datetime.today()} URL: {main_URL}"

    # logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    print(logText)

    # get all the documents and their details
    scrapeDF = MDR_Scrapping(main_URL, config.SeleniumDocker)

    logText = f"Website data captured at {datetime.today()}"
    # logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    print(logText)

    t1 = time.time()
    update_list, download_list = check_for_new_documents(
        config, mysql_driver, scrapeDF, scrapeURLId, scrapeScript
    )
    print("final download  list size:")
    print(len(download_list))

    logText = f"File downloading completed at {datetime.today()} time taken: {str((time.time() - t1))}"
    # logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    print(logText)

    #     # handle cancelled documents
    #     cancel_list = check_for_cancelled_documents(mysql_driver, DateToday, scrapeURLId)

    #     logText = f'Number of cancelled documents since last scrape: {len(cancel_list)}'
    #     logList.append(logText)
    #     scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    #     print(logText)

    #     # Mark the newly cancelled documents in the documents table
    #     if cancel_list and len(cancel_list) > 0:
    #         cancel_documents(mysql_driver, cancel_list)

    #     #save_log_data_to_db(logList, mysql_driver)
    print("Scaped URL:" + main_URL)


if __name__ == "__main__":
    try:
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
