## Library import
import logging
import os
import re
import traceback
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests
from common_tools.log_config import configure_logging_from_argv
from database.document_service import (
    delete_document,
    get_document_by_title,
    insert_document,
)
from database.entity.Document import Document
from database.entity.ScriptsProperty import ScriptsConfig, parseCredentialFile
from database.utils.MySQLFactory import MySQLDriver
from database.utils.WebDriverFactory import WebDriverFactory
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.ie.webdriver import WebDriver
from selenium.webdriver.support.ui import WebDriverWait

from packages.scraping.src.scraping.FDA_CFR import parse_remaining_args

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def scrap_table(driver, url):
    i = 2
    cn = 0
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
    while True:
        try:
            tr = url + f"[{i}]"
            driver.find_element_by_xpath(tr)
            td = [
                driver.find_element_by_xpath(tr + "/" + "td" + f"[{i}]").text
                for i in range(1, 4)
            ]
            dataset["title"].append(td[1])

            link = driver.find_element_by_xpath(tr + "/" + "td[2]/a").get_attribute(
                "href"
            )
            dataset["pdf_file_url"].append(link)

            file_name = link.split("/")[-1]
            log.debug("Processing file: %s", file_name)
            dataset["pdf_file_name"].append(file_name)

            dataset["description"].append(td[0])
            dataset["filetype"].append(5)
            dataset["Download"].append(True)
            dataset["status"].append("Active")
            dataset["active_date"].append(date(datetime.today().year, 1, 1))
            dataset["Number"].append(cn)

            i += 1
            cn += 1
        except Exception as e:
            log.error("Error scraping table: %s", e)
            traceback.print_exc()
            break
    df = pd.DataFrame(dataset)
    return df


def wait_for_page_load(driver):
    WebDriverWait(driver, 30).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )


class DocumentLink:
    name: str
    date: date
    link: str
    doc_type: str  # html or pdf

    def __init__(self, name: str, date: date, link: str, doc_type: str):
        self.name = name
        self.date = date
        self.link = link
        self.doc_type = doc_type

    def get_file_name(self) -> str:
        return clean_filename(f"{self.name}.{self.doc_type}")


def get_document_link_from_page(driver: WebDriver, url: str) -> Tuple[str, str]:
    driver.get(url)
    wait_for_page_load(driver)
    html_download_link = driver.find_element(By.ID, "format_language_table_HTML_EL")
    pdf_download_link = driver.find_element(By.ID, "format_language_table_PDF_EL")
    return html_download_link.get_attribute("href"), pdf_download_link.get_attribute(
        "href"
    )


def make_driver(headless=True, log_path="chromedriver.log"):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    # Stable flags for servers/containers
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--remote-debugging-port=9222")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-gpu")

    # If Chromium is used, set binary explicitly (optional if Google Chrome)
    for cand in (
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ):
        if os.path.exists(cand):
            opts.binary_location = cand
            break

    # Ensure localhost isn’t proxied (can cause weird timeouts)
    for k in ("NO_PROXY", "no_proxy"):
        os.environ[k] = "localhost,127.0.0.1"

    service = Service(log_output=log_path)  # captures ChromeDriver logs
    # If you installed chromedriver into a non-PATH location, pass executable_path=...
    # service = Service(executable_path="/usr/local/bin/chromedriver", log_output=log_path)

    drv = webdriver.Chrome(options=opts, service=service)
    return drv


def get_document_list(
    docker_url: str = "http://localhost:4444/wd/hub",
) -> List[DocumentLink]:
    log.info(f"Getting document list from docker_url: {docker_url}")
    main_url = "http://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02017R0745-20250110"
    try:
        log.info(
            "Starting get_document_list docker_url=%s main_url=%s", docker_url, main_url
        )
        driver = None
        driver = WebDriverFactory.getWebDriverInstance(
            browser="docker", docker_url=docker_url
        )
        log.info("Initialized WebDriver via Docker at %s", docker_url)
        # navigate to the main_URL
        driver.get(main_url)
        wait_for_page_load(driver)
        log.debug("Loaded main URL and page is ready")

        # look for table with all the amendments. The tables have no ids, so look for
        # a specific link and then get the table from the parent element
        amendment_link = (
            "https://eur-lex.europa.eu/legal-content/EN/AUTO/?uri=celex:32020R0561"
        )
        links_elements = driver.find_elements(By.XPATH, "//a")
        log.debug("Total <a> elements found: %d", len(links_elements))
        amendment_element = None
        for link_element in links_elements:
            link_url = link_element.get_attribute("href")

            if link_url == amendment_link:
                amendment_element = link_element
                break

        if amendment_element is None:
            log.error("Amendment element not found for link: %s", amendment_link)
            return []
        else:
            log.info("Amendment element found for link: %s", amendment_link)

        # get parent table element
        page_links: List[DocumentLink] = []
        table_element = amendment_element.find_element(By.XPATH, "./ancestor::table")
        rows = table_element.find_elements(By.TAG_NAME, "tr")
        log.debug("Amendment table rows found: %d", len(rows))
        for row in rows:
            # Get all cells (both <td> and <th>) in the current row
            cells = row.find_elements(By.XPATH, ".//th | .//td")
            [cell.text.strip() for cell in cells]
            if len(cells) != 5:
                continue
            _, link_cell, _, _, date = cells

            # get document name, link and date and append to list
            doc_name = link_cell.text.strip()
            doc_link = link_cell.find_element(By.TAG_NAME, "a").get_attribute("href")
            doc_date = datetime.strptime(date.text, "%d.%m.%Y").date()
            page_links.append(DocumentLink(doc_name, doc_date, doc_link, "html"))

        log.info("Collected amendment page links: %d", len(page_links))
        doc_links: List[DocumentLink] = []
        for page_link in page_links:
            log.debug(
                "Fetching downloadable links for: %s (%s)",
                page_link.name,
                page_link.link,
            )
            html_link, pdf_link = get_document_link_from_page(driver, page_link.link)

            doc_links.append(
                DocumentLink(
                    f"{page_link.name}.html", page_link.date, html_link, "html"
                )
            )
            doc_links.append(
                DocumentLink(f"{page_link.name}.pdf", page_link.date, pdf_link, "pdf")
            )

        log.info("Total downloadable links prepared: %d", len(doc_links))
        return doc_links

    except Exception as e:
        # close the driver
        if driver is not None:
            driver.close()
        log.error("Error getting document list: %s", e)
        traceback.print_exc()
        return []


def clean_filename(file_name: Path | str) -> str:
    """
    Clean a filename by removing or replacing characters not allowed in file systems.

    Args:
        file_name (Path): The file name (with or without extension).

    Returns:
        str: A sanitized version of the filename safe for most file systems.
    """
    # Get the stem and extension
    if isinstance(file_name, str):
        file_name = Path(file_name)
    stem = file_name.stem
    suffix = file_name.suffix

    # Define invalid characters for Windows and other OSes
    # Windows forbids: < > : " / \ | ? *
    # Also remove control chars and trim spaces/dots
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", stem)
    cleaned = cleaned.strip(" .")

    # Replace multiple spaces with a single space
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()

    # Fallback for empty filenames
    if not cleaned:
        cleaned = "untitled"

    return cleaned + suffix


def upload_document(
    doc: DocumentLink, mysql_driver: MySQLDriver, path_to_save: Path | str
):
    document_type = (
        3  # EU MDR as defined https://meddev.enginius.ai/#/admin/document-type
    )
    now_date = datetime.today().date()
    file_name = doc.get_file_name()
    doc_path = path_to_save / file_name
    success = download_file(doc.link, doc_path)

    if not success:
        log.error("Failed to download file: '%s'", doc.link)
        return

    document = Document(
        number="",
        title=doc.name,
        description=doc.name,
        url=doc.link,
        documentType=document_type,
        documentStatus=1,
        activeDate=doc.date,
        modifiedDate=doc.date,
        lastScrapeDate=now_date,
        createdDate=now_date,
    )
    insert_document(mysql_driver, document)


def download_file(url: str, path_to_save: Path | str) -> bool:
    """Download a file from a given URL to a given path.

    Args:
        url (str): The URL of the file to download.
        path_to_save (Path | str): The path to save the file to.

    Returns:
        bool: True if the file was successfully downloaded, False otherwise.
    """
    if isinstance(path_to_save, str):
        path_to_save = Path(path_to_save)
    if not path_to_save.exists():
        path_to_save.parent.mkdir(parents=True, exist_ok=True)
    hdr = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
        #'Accept-Encoding': 'gzip, deflate, br',
        "Referer": "https://www.medical-device-regulation.eu/",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    response = requests.get(
        url,
        headers=hdr,
        verify=True,  # Use default certificate bundle
        stream=True,
    )
    # response = requests.get(url,  headers=hdr, verify = '/etc/ssl/cert.pem', stream=True)
    # isolate PDF filename from URL

    if response.status_code == 200:
        with open(path_to_save, "wb") as pdf_object:
            pdf_object.write(response.content)
            log.debug("%s was successfully saved!", path_to_save)
            return True
    log.error("Failed to download file: %s", url)
    log.error("HTTP response status code: %s", response.status_code)
    return False


def run(config: ScriptsConfig, scrapeURLId: int):
    mysql_driver = MySQLDriver(cred=config.databaseConfig.__dict__)

    # Get all the documents currently in the database
    doc_list: List[DocumentLink] = get_document_list(docker_url=config.SeleniumDocker)
    if len(doc_list) == 0:
        log.error("No documents found")
        return

    path_to_save = Path(config.downloadDir)
    for doc in doc_list:
        log.debug(f"Document: {doc.name} date: {doc.date} link: {doc.link}")
        existing_doc = get_document_by_title(mysql_driver, doc.name)

        if existing_doc is not None:
            existing_doc_date = existing_doc.activeDate
            if not (path_to_save / doc.get_file_name()).exists():
                log.debug(f"Document {doc.name} does not exist")
                success = download_file(doc.link, path_to_save / doc.get_file_name())
                if not success:
                    log.error(f"Failed to download file: {doc.name}")
                    continue
            if doc.date <= existing_doc_date:
                log.debug(
                    f"Document {doc.name} already exists and is newer than the existing document"
                )
                continue
            else:
                log.debug(
                    f"Document {doc.name} already exists and is older than the existing document"
                )
                delete_document(mysql_driver, existing_doc.documentId)
                upload_document(doc, mysql_driver, path_to_save)
        else:
            log.debug(f"Document {doc.name} does not exist")
            upload_document(doc, mysql_driver, path_to_save)
        continue


if __name__ == "__main__":
    from database.configs import config_dir

    remaining_args = configure_logging_from_argv(default_level="INFO")
    repo_id, docIdsList = parse_remaining_args(remaining_args)

    configs = parseCredentialFile(config_dir / "dev_test_tlp_config.json")
    download_dir = Path(__file__).parent / "data" / "EU_MDR_Ammendement"
    download_dir.mkdir(parents=True, exist_ok=True)
    configs.downloadDir = str(download_dir)
    if configs:
        run(configs)
    else:
        log.error("No configuration file found")
        exit(1)
