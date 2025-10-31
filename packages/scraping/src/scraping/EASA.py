import io
import logging
import os
import re
import sys
import time
import traceback
import zipfile
from datetime import datetime
from multiprocessing import current_process

import pandas as pd
from database.document_service import (
    find_document_by_url,
    find_documents_not_scraped_on_date,
    get_parsing_script_by_document_type_name,
    get_scrape_script_by_scraperUrlId,
    insert_documents_bulk2,
)
from database.entity.Document import Document
from database.entity.ScrapScript import ScrapScript
from database.entity.ScriptsProperty import ScriptsConfig, parseCredentialFile
from database.scrape_url_service import scrape_url_append_log
from database.utils.MySQLFactory import MySQLDriver

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Console (stdout) handler
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)

from common_tools.log_config import configure_logging_from_argv
from database.utils.util import get_dir_safe
from database.utils.WebDriverFactory import WebDriverFactory
from PyPDF2 import PdfReader

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

logList = []
DateToday = datetime.today()


def clean_string(input_string):
    input_string = input_string.strip()

    # Define the pattern to remove unwanted characters
    # pattern2 = r'\([^)]*\)'

    # input_string = re.sub(pattern2, '', input_string)

    pattern = r"[^a-zA-Z0-9\s]"

    # Use the pattern to remove unwanted characters
    cleaned_string = re.sub(pattern, "_", input_string)

    return cleaned_string


def emptydir(top):
    if top == "/" or top == "\\":  # don't delete root folder -> could be disaster
        return
    else:
        log.debug("Emptying folder: %s", top)
        for root, dirs, files in os.walk(top, topdown=False):
            for name in files:
                log.debug("Removing file: %s", os.path.join(root, name))
                os.remove(os.path.join(root, name))
            for name in dirs:
                log.debug("Removing folder: %s", os.path.join(root, name))
                os.rmdir(os.path.join(root, name))


def extract_xml_from_zip_file(zip_file, download_dir):
    # dir_name = '/home/dshah/Inspird-2023-dev/Web_Scrapping/Easa_Reg_Docs'
    dir_name = download_dir

    temp_extract_path = get_dir_safe(download_dir + "/temp_extract")

    try:
        os.lstat(zip_file)
        if zip_file.endswith("zip"):
            # with zipfile.ZipFile(os.path.join(dir_name,zip_file), 'r') as zip_ref:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(temp_extract_path)

                xml_file_path = os.path.join(dir_name, zip_file.split(".")[0] + ".xml")
                os.rename(
                    os.path.join(temp_extract_path, os.listdir(temp_extract_path)[0]),
                    xml_file_path,
                )

                # os.remove(os.listdir(temp_extract_path))
                emptydir(temp_extract_path)
                #
                return True, xml_file_path
    except OSError:
        log.debug("Zipfile %s does not exist or is not accessible!", zipfile)
        traceback.print_exc()

    return False, ""


def download_file(URL, file_path):
    hdr = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9,gu;q=0.8,hi;q=0.7",
        "Connection": "keep-alive",
    }
    chunk_size = (1024 * 1024) * 1  # 1MB

    # response = requests.get(URL, headers=hdr,verify = '/etc/ssl/cert.pem')
    response = requests.get(
        URL,
        headers=hdr,
        verify="/app/scripts/venv/lib/python3.9/site-packages/certifi/cacert.pem",
    )
    total = int(response.headers.get("content-length", 0))
    filename = request.urlopen(request.Request(URL)).info().get_filename()
    extension = filename.split(".")[-1]
    if extension.lower() == "zip":
        # file_base_name, file_extension = os.path.splitext(filename)
        file_path = os.path.join(file_path, filename)
        log.debug("Saving file to: %s", file_path)
    log.debug("File extension: %s", extension)
    #    file_path = file_path+'/'+clean_string(title)+'.'+extentions
    with open(file_path, "wb") as file, tqdm(
        desc=file_path,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
    return file_path, extension


def download_files(config: ScriptsConfig, document: Document, pdf_url, xml_url):
    try:
        download_dir = get_dir_safe(config.downloadDir)
        pdf_file_name = document.pdfFileName
        xml_file_name = document.sourceFileName
        pdf_file_path = download_dir + "/" + clean_string(pdf_file_name) + ".pdf"
        xml_file_path = download_dir + "/" + clean_string(xml_file_name) + ".xml"
        log.debug(
            "%s -> Downloading file URL: %s to location: %s",
            current_process(),
            document.url,
            pdf_file_path,
        )
        time.sleep(10)

        # download PDF
        if pdf_url and len(pdf_url) > 0:
            _, extension = download_file(pdf_url, pdf_file_path)
            document.pdfFileName = os.path.relpath(pdf_file_path, config.downloadDir)

        # download XML
        if xml_url and len(xml_url) > 0:
            xml_file_path, extension = download_file(xml_url, download_dir)
            if extension.lower() == "zip":
                retval, xml_file_path2 = extract_xml_from_zip_file(
                    xml_file_path, download_dir
                )
                if retval is True:
                    document.sourceFileName = os.path.relpath(
                        xml_file_path2, config.downloadDir
                    )
                else:
                    document.sourceFileName = os.path.relpath(
                        xml_file_path, config.downloadDir
                    )
        else:
            document.sourceFileName = ""
        document.fileSize = os.path.getsize(pdf_file_path)
        document.parsed = False

        return document, "Success"
    except Exception as ex:
        log.debug("File download failed: %s", document.title + ".pdf")
        log.debug("%s", ex)
        traceback.print_exc()
        return None, str(traceback.format_exc())


def download_link(driver2, url):
    driver2.get(str(url))
    links = []
    filenames = []
    for i in driver2.find_elements_by_class_name("matomo_download"):
        name = i.text
        link = i.get_attribute("href")
        links.append(link)
        filenames.append(name)

    return links[:3], filenames[:3]


def EASA_Scrapping(url, docker_url):
    # driver = WebDriverFactory.getWebDriverInstance(browser='chrome')
    driver = WebDriverFactory.getWebDriverInstance(
        browser="docker", docker_url=docker_url
    )
    # driver2 = WebDriverFactory.getWebDriverInstance(browser='chrome')
    driver2 = WebDriverFactory.getWebDriverInstance(
        browser="docker", docker_url=docker_url
    )
    # url = 'https://www.easa.europa.eu/en/document-library/easy-access-rules?page=4'
    driver.get(url)
    all_list = driver.find_elements_by_xpath(
        '//*[@id="block-easa-foundation-system-main"]/div/div/div/div/div/ul/li'
    )
    dataset = {
        "Number": [],
        "title": [],
        "discription": [],
        "pdf_file_name": [],
        "pdf_file_url": [],
        "filetype": [],
        "status": [],
        "active_date": [],
        "Download": [],
    }
    cn = 0
    for i in all_list:
        try:
            t = i.find_element_by_tag_name("a")

            data = i.text.split("\n")

            link = t.get_attribute("href")

            if len(data) > 2:
                dataset["title"].append(data[2])

                dataset["active_date"].append(data[0])
            else:
                dataset["title"].append(data[1])

                dataset["active_date"].append(data[0])

            file_links, file_names = download_link(driver2, link)

            dataset["pdf_file_url"].append(file_links)

            dataset["pdf_file_name"].append(file_names)

            dataset["discription"].append("")

            dataset["filetype"].append(6)

            dataset["Download"].append(True)

            dataset["status"].append("Active")

            dataset["Number"].append(cn)

            cn += 1
        except Exception:
            if "SUPERSEDED’" in data[1]:
                log.debug("Dataset content: %s", dataset)
                cn = cn + 1
                link = str(link).replace("-superseded", "")
                if len(data) > 2:
                    dataset["title"].append(data[2])
                    dataset["active_date"].append(data[0])
                else:
                    dataset["title"].append(data[1])
                    dataset["active_date"].append(data[0])

                pdf_file_link, pdf_file_name = download_link(driver2, link)
                dataset["pdf_file_url"].append(pdf_file_link)

                dataset["pdf_file_name"].append(pdf_file_name[0])

                dataset["discription"].append("")

                dataset["filetype"].append(6)

                dataset["Download"].append(True)

                dataset["status"].append("Active")
                dataset["Number"].append(cn)

    df = pd.DataFrame.from_dict(dataset, orient="index").T
    df.iloc[-2]["pdf_file_url"] = df.iloc[-1]["pdf_file_url"]
    df = df[:-1]
    return df


def check_for_new_documents(
    mysql_driver, scrapeDF, scrapeURLId, scrapeScript: ScrapScript
):
    # global logList
    download_list = []
    update_list = []

    pdf_urls = []
    xml_urls = []
    for idx, row in scrapeDF.iterrows():
        file_urls = row["pdf_file_url"]

        file_names = row["pdf_file_name"]
        pdf_file_url = []
        pdf_file_name = ""
        xml_file_url = []
        xml_file_name = ""

        for i, fname in enumerate(file_names[:2]):
            if "pdf" in fname.lower():
                pdf_file_url = file_urls[i]
                pdf_file_name = fname
            elif "xml" in fname.lower():
                xml_file_url = file_urls[i]
                xml_file_name = fname

        try:
            #            if row['Number'] == 0:
            #                continue

            if isinstance(pdf_file_url, list):
                if len(pdf_file_url) > 0:
                    file_url = str(pdf_file_url[0])
                else:
                    #                    skip_list.append(str(row['Number']) + ' ' + row['title'])
                    logText = (
                        "Skipped File "
                        + row["title"]
                        + " with AC at "
                        + " URL = <null>  on "
                        + str(datetime.today())
                    )
                    # logList.append(logText)
                    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                    print(logText)
                    continue

            docInDB = find_document_by_url(mysql_driver, pdf_file_url)

        except Exception:
            logText = f"Failed for : {pdf_file_url} \n"
            logText += traceback.format_exc()
            # logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            print(logText)
            continue

        # scrapeScript: ScrapScript = get_scrap_script_by_file_name(mysql_driver, os.path.basename(__file__))

        if not docInDB:
            print(f"{pdf_file_url} not found in DB")

            try:
                document = Document(
                    number=row["Number"],
                    title=row["title"],
                    description=row["discription"][:1024],
                    url=str(pdf_file_url),
                    # documentType=row['filetype'],
                    documentType=scrapeScript.documentTypeID,
                    documentStatus=1,  # Active
                    activeDate=datetime.strptime(row["active_date"], "%d %b %Y"),
                    inactiveDate=None,
                    sourceFileName=xml_file_name,
                    pdfFileName=pdf_file_name,
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
                    # scrapingLog = 'scraped successfully'
                    sourceProject=0,
                )
                download_list.append(document)
                pdf_urls.append(pdf_file_url)
                xml_urls.append(xml_file_url)

                print(type(document))
                # insert_document(mysql_driver, document)

                logText = (
                    str(row["pdf_file_name"])
                    + " with filetype = "
                    + str(row["filetype"])
                    + " at "
                    + pdf_file_url
                    + " downloaded on "
                    + str(datetime.today().date())
                )
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            except Exception:
                logText = f"New Document row creation failer for : {file_url} \n"
                logText += traceback.format_exc()
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                print(logText)

        else:
            print(f"{pdf_file_url} found in DB")
            docInDB.lastScrapeDate = datetime.today().date()
            logText = (
                "File "
                + row["title"]
                + " with Guidance Doc at "
                + pdf_file_url
                + " scraped and not changed on "
                + str(datetime.today())
            )
            logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            print(logText)
            update_list.append(docInDB)

    return update_list, download_list, pdf_urls, xml_urls


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

    db_name = config.databaseConfig.__dict__["database"]
    sql_stmt = f"SELECT parsingScriptID from {db_name}.parsingscripts where title like '%PDF%';"
    result = get_parsing_script_by_document_type_name(mysql_driver, sql_stmt)

    for r in result:
        PDF_parsingScriptID = r[0]
        break

    main_URL = "https://www.easa.europa.eu/en/document-library/easy-access-rules?page=4"

    logText = f"ScrapeID: {scrapeURLId}: Scrape for EASA Regulations Documents started at {datetime.today()} URL: {main_URL}"

    # logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    print(logText)

    # get all the documents and their details
    scrapeDF = EASA_Scrapping(main_URL, docker_url=config.SeleniumDocker.rstrip("/"))

    logText = f"Website data captured at {datetime.today()}"
    # logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    print(logText)
    # list_of_urls = list(scrapeDF['pdf_file_url'])
    # title = list(scrapeDF['title'])
    # download_list_link = [item for sublist in list_of_urls for item in sublist]

    update_list, download_list, pdf_urls, xml_urls = check_for_new_documents(
        mysql_driver, scrapeDF, scrapeURLId, scrapeScript
    )

    # #update the unchanged documents in DB (just set their lastScrapeDate = DateToday)
    # if update_list and len(update_list) > 0:
    #     update_documents(mysql_driver, update_list)

    print("final download  list size:")
    print(download_list)
    print(len(download_list))

    bulk_document = []
    logText = f"File downloading started at {datetime.today()}"
    # logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)

    t1 = time.time()
    print("starting download files")
    cnt = 0
    for i, doc in enumerate(download_list):
        # print("DOC:", str(doc.number))
        cnt += 1

        updated_doc, msgText = download_files(config, doc, pdf_urls[i], xml_urls[i])
        if updated_doc is not None:
            # if sourceFileName is not present, then parse the PDF file
            if (
                updated_doc.sourceFileName is None
                or len(updated_doc.sourceFileName) == 0
            ):
                updated_doc.parsingScriptID = PDF_parsingScriptID
            bulk_document.append(updated_doc)
        else:
            logList.append(msgText)
            scrape_url_append_log(mysql_driver, scrapeURLId, msgText)

    logText = f"File downloading completed at {datetime.today()} time taken: {str((time.time() - t1))}"
    logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    print(logText)

    docIDsList = None
    try:
        if bulk_document is not None and len(bulk_document) > 0:
            docIDsList = insert_documents_bulk2(mysql_driver, bulk_document)

        if docIDsList is not None and len(docIDsList) > 0:
            logText = f"Newly added docIDs: {docIDsList}"
            # logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            print(logText)
    except:
        logText = traceback.format_exc()
        # logList.append(logText)
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)
        print(logText)

    # # handle cancelled documents
    # cancel_list = check_for_cancelled_documents(mysql_driver, DateToday, scrapeURLId)

    # logText = f'Number of cancelled documents since last scrape: {len(cancel_list)}'
    # logList.append(logText)
    # scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    # print(logText)

    # # Mark the newly cancelled documents in the documents table
    # if cancel_list and len(cancel_list) > 0:
    #     cancel_documents(mysql_driver, cancel_list)

    logText = f"Scraping URL: {main_URL} DONE"
    # logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    print(logText)

    print("downloaded")


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
