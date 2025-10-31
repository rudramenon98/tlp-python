import logging
import os
import traceback
from datetime import date, datetime
from multiprocessing import current_process
from urllib.request import urlretrieve

import lxml.html
import requests
from app.document_service import (
    get_scrape_script_by_scraperUrlId,
    insert_document,
)
from app.entity.Document import Document
from app.entity.ScrapScript import ScrapScript
from logconfig import configure_logging_from_argv

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
"""
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Console (stdout) handler
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
"""

from app.entity.ScriptsProperty import ScriptsConfig, parseCredentialFile
from app.scrape_url_service import (
    scrape_url_append_log,
)
from app.utils.MySQLFactory import MySQLDriver
from app.utils.util import get_dir_safe

# chrome_options = Options()
# chrome_options.add_argument("user-agent=whatever you want")
# chrome_options.add_argument("--disable-extensions")
# chrome_options.add_argument("--disable-gpu")
# chrome_options.add_argument("--no-sandbox") # linux only
# chrome_options.add_argument('--no-sandbox')
# chrome_options.add_argument('--disable-dev-shm-usage')
# chrome_options.add_argument("--headless")
DefaultActiveDate = date(date.today().year, 1, 1)

"""This Script is used for Scrapping MDR Documents"""


def get_pdf_html(path_to_save, type="MDR_EN"):
    """Link for Europian Regulations"""
    # response  = urllib.request.urlopen('https://www.medical-device-regulation.eu/download-mdr/')
    # html_string = response.read()
    hdr = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
        #'Accept-Encoding': 'gzip, deflate, br',
        "Referer": "https://www.medical-device-regulation.eu/",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }

    response = requests.get(
        "https://www.medical-device-regulation.eu/download-mdr/",
        headers=hdr,
        verify="/app/scripts/venv/lib/python3.9/site-packages/certifi/cacert.pem",
    )
    # response = requests.get('https://www.medical-device-regulation.eu/download-mdr/', headers=hdr, verify = '/etc/ssl/cert.pem')
    html_string = response.text

    # Parse the HTML content using lxml
    tree = lxml.html.fromstring(html_string)

    ### Getting link for HTML and PDF
    for i in tree.iter():
        if i.tag == "td":
            for j in i:
                if j.text == type:
                    link = j.attrib["href"]
                    if link.endswith(".pdf"):
                        pdf = j.attrib["href"]
                    else:
                        html = j.attrib["href"]

    URL_pdf = pdf
    html_url, pdf_url = download_pdf_file(URL_pdf, html, type, path_to_save)

    log.debug("Download is Done")
    return html_url, pdf_url


def download_pdf_file(url: str, html_page_url, type, path_to_save) -> bool:
    """Download PDF from given URL to local directory.

    :param url: The url of the PDF file to be downloaded
    :return: True if PDF file was successfully downloaded, otherwise False.
    """
    ##downaloading html page url

    html_filename, headers = urlretrieve(
        html_page_url, filename=path_to_save + type + ".html"
    )
    pdf_url = url

    hdr = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
        #'Accept-Encoding': 'gzip, deflate, br',
        "Referer": "https://www.medical-device-regulation.eu/",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }

    # Request URL and get response object
    response = requests.get(
        url,
        headers=hdr,
        verify="/app/scripts/venv/lib/python3.9/site-packages/certifi/cacert.pem",
        stream=True,
    )
    # response = requests.get(url,  headers=hdr, verify = '/etc/ssl/cert.pem', stream=True)
    # isolate PDF filename from URL
    pdf_file_name = os.path.basename(url)
    if response.status_code == 200:
        # Save in current working directory
        # filepath = os.path.join(os.getcwd(), pdf_file_name)
        filepath = path_to_save + pdf_file_name
        with open(filepath, "wb") as pdf_object:
            pdf_object.write(response.content)
            log.debug("%s was successfully saved!", pdf_file_name)
            return html_page_url, pdf_url
    else:
        log.debug("Uh oh! Could not download %s,", pdf_file_name)
        log.debug("HTTP response status code: %s", response.status_code)
        return None, None


def download_file(config: ScriptsConfig, type="MDR_EN"):
    try:
        mysql_driver = MySQLDriver(cred=config.databaseConfig.__dict__)
        # file_name = document.additionalInfo + ".pdf"
        # dir = config.rootDataDir + config.downloadDirInRootDir
        dir = config.downloadDir
        path_to_save = get_dir_safe(dir) + "/"
        log.debug(
            "%s -> Downloading file  on location:%s", current_process(), path_to_save
        )
        html_url, pdf_url = get_pdf_html(path_to_save, type)
        # response = urllib.request.urlopen(document.url)
        # get_pdf_html()
        # document.pdfFileName = os.path.relpath(path, config.rootDataDir)
        # document.fileSize = os.path.getsize(path)
        # document.parsed = False
        # insert_document(mysql_driver, document)
        mysql_driver.close()
        return html_url, pdf_url
    except Exception as ex:
        log.debug("%s", ex)
        traceback.print_exc()
        return None, None


def run(config: ScriptsConfig, scrapeURLId):

    # scrapeURLId = 1  currently hard-coded
    mysql_driver = MySQLDriver(cred=config.databaseConfig.__dict__)

    logText = "MDR Document Scraping : started"
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    log.debug("%s", logText)

    html_url, pdf_url = download_file(config)

    logText = "MDR Document Scraping : files downloaded"
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    log.debug("%s", logText)

    scrapeScript: ScrapScript = get_scrape_script_by_scraperUrlId(
        mysql_driver, scrapeURLId
    )

    # dir = config.rootDataDir + config.downloadDirInRootDir
    dir = config.downloadDir
    path_to_save = get_dir_safe(dir) + "/"

    src_filename = path_to_save + "MDR_EN" + ".html"
    src_filename = os.path.relpath(src_filename, config.downloadDir)
    url_path = pdf_url
    pdf_filename = path_to_save + os.path.basename(pdf_url)
    fileSize = os.path.getsize(pdf_filename)
    pdf_filename = os.path.relpath(pdf_filename, config.downloadDir)

    try:
        document = Document(
            number="",
            title="EU MDR Regulation",
            description="EU MDR Regulation",
            url=url_path,
            documentType=scrapeScript.documentTypeID,
            documentStatus=1,  # Active
            # activeDate=None,
            activeDate=DefaultActiveDate,
            inactiveDate=None,
            sourceFileName=src_filename,
            pdfFileName=pdf_filename,
            parsed=False,
            embedded=False,
            parsingScriptID=scrapeScript.defaultParsingScriptID,
            createdDate=datetime.today().date(),
            modifiedDate=datetime.today().date(),
            # additionalInfo=None,
            fileSize=fileSize,
            parsingLog="notParsed",
            embeddingLog="notEmbedded",
            noOfParagraphs=0,
            lastScrapeDate=datetime.today().date(),
            sourceProject=0,
        )
        insert_document(mysql_driver, document)
        logText = (
            pdf_filename
            + " with filetype = MDR "
            + " at "
            + url_path
            + " downloaded on "
            + str(datetime.today().date())
        )
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)
        log.debug("%s", logText)
    except Exception:
        logText = (
            pdf_filename
            + " with filetype = MDR "
            + " at "
            + url_path
            + " downloaded failed on "
            + str(datetime.today().date())
        )
        logText += traceback.format_exc()
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)
        log.debug("%s", logText)

        # else:
        #     docInDB.lastScrapeDate = datetime.today().date()
        #     # docInDB.scrapingLog = 'File ' + row['title']  + ' with AC at ' + row['pdf_file_url'] + ' scraped and not changed on ' + str(datetime.today())
        #     logText = 'File ' + row['title'] + ' with AC at ' + row[
        #         'pdf_file_url'] + ' scraped and not changed on ' + str(datetime.today())
        #     logList.insert(0, logText)
        #     update_list.append(docInDB)

    logText = f"MDR Document Downloaded: {pdf_filename} completed"
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    log.debug("%s", logText)

    print("Download  MDR  documents")


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
        # configs = parseCredentialFile('//dockers/Enginius/test/scripts/testmed-tlp_config.json')

        if configs:
            run(configs, scrapeURLId)
    except Exception as e:
        print("The EXCEPTION >>>>>>>>>>>>>> ")
        print(traceback.format_exc())
        traceback.print_exc()
        print(e)
