import logging
import os
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import current_process

import requests
from database.document_service import (
    find_document_by_url,
    get_scrape_script_by_scraperUrlId,
    insert_documents_bulk2,
)
from database.entity.Document import Document
from database.entity.ScrapScript import ScrapScript
from database import CONFIG_DIR
from database.entity.ScriptsProperty import ScriptsConfig, parseCredentialFile
from database.scrape_url_service import scrape_url_append_log
from database.utils.MySQLFactory import MySQLDriver
from database.utils.util import get_dir_safe
from database.utils.WebDriverFactory import WebDriverFactory
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Configure logging to suppress all output
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("selenium").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("webdriver_manager").setLevel(logging.ERROR)

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
        file_name = document.title[:50].replace("/", " ") + ".pdf"
        dir_name = config.downloadDir
        path = get_dir_safe(dir_name) + "/" + file_name
        print(
            f"{current_process()} -> Downloading file {sequence} url:{document.url} on location:{path}"
        )
        download_file_low(document.url, path)
        document.pdfFileName = os.path.relpath(path, config.downloadDir)
        document.fileSize = os.path.getsize(path)
        document.parsed = False
        document.embedded = False
        return document, "Success"
    except Exception as ex:
        print("File is not Downloaded:", document.title + ".pdf")
        print(ex)
        traceback.print_exc()
        return None, str(traceback.format_exc())


def get_main_page(config: ScriptsConfig):
    # driver = WebDriverFactory.getWebDriverInstance(browser='chrome')
    driver = WebDriverFactory.getWebDriverInstance(
        browser="docker", docker_url=config.SeleniumDocker
    )
    WebDriverWait(driver, 15)

    # Navigate to the NIST search page
    driver.get("https://csrc.nist.gov/publications/search")
    WebDriverWait(driver, 20)
    time.sleep(20)

    return driver


def search_series(driver, series_names):
    """Search for publications in specific series"""
    wait = WebDriverWait(driver, 10)

    # Wait for the page to load and checkboxes to be present
    wait.until(
        EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "input[name='series-lg']")
        )
    )

    # First, uncheck all series checkboxes
    all_series = driver.find_elements(By.CSS_SELECTOR, "input[name='series-lg']")
    for checkbox in all_series:
        if checkbox.is_selected():
            checkbox.click()

    # Then check only the requested series
    for series_name in series_names:
        try:
            series_checkbox = wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, f"input[name='series-lg'][value='{series_name}']")
                )
            )
            if not series_checkbox.is_selected():
                series_checkbox.click()
        except Exception as e:
            print(
                f"Warning: Could not find or click checkbox for series '{series_name}': {str(e)}"
            )

    # Click search button
    search_button = wait.until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button#pubs-search-submit-lg"))
    )
    search_button.click()

    # Wait for results to load
    wait.until(
        EC.presence_of_all_elements_located(
            (By.XPATH, "//a[contains(@href, '/pubs/') and not(contains(@href, 'ipd'))]")
        )
    )


def process_page(config: ScriptsConfig, driver, mysql_driver, scrapeURLId):
    page_num = 1
    batch_size = 5  # Process 5 links at a time
    download_list = []
    wait = WebDriverWait(driver, 10)

    scrapeScript: ScrapScript = get_scrape_script_by_scraperUrlId(
        mysql_driver, scrapeURLId
    )

    while True:
        try:
            print(f"\nProcessing page {page_num}")

            # Wait for publication links to be present
            try:
                wait.until(
                    EC.presence_of_all_elements_located(
                        (
                            By.XPATH,
                            "//a[contains(@href, '/pubs/') and not(contains(@href, 'ipd')) and normalize-space(text())]",
                        )
                    )
                )
            except TimeoutException:
                print(
                    "Timeout waiting for publication links. The page might be empty or not loaded properly."
                )
                break

            # Find all publication links
            publication_links = driver.find_elements(
                By.XPATH,
                "//a[contains(@href, '/pubs/') and not(contains(@href, 'ipd')) and normalize-space(text())]",
            )

            if not publication_links:
                print("No publication links found on this page")
                break

            # Store the main window handle
            main_window = driver.current_window_handle

            # Process links in batches
            for i in range(0, len(publication_links), batch_size):
                batch = publication_links[i : i + batch_size]

                # Open links in new tabs
                for link in batch:
                    try:
                        href = link.get_attribute("href")
                        text = link.text.strip()
                        if not text:  # Skip links with no text
                            continue

                        # Open link in new tab
                        driver.execute_script("window.open(arguments[0]);", href)
                    except Exception as e:
                        print(f"Error opening link: {str(e)}")
                        continue

                # Process all opened tabs in this batch
                for window_handle in driver.window_handles[1:]:  # Skip the main window
                    try:
                        # Switch to the tab
                        driver.switch_to.window(window_handle)

                        # Get the title from the page
                        page_title = driver.title.strip()
                        print(f"Page title: {page_title}")

                        # Wait for PDF link to be available
                        try:
                            # Find all PDF links on the page
                            pdf_links = wait.until(
                                EC.presence_of_all_elements_located(
                                    (By.ID, "pub-local-download-link")
                                )
                            )

                            if not pdf_links:
                                continue

                            # Download each PDF
                            for pdf_link in pdf_links:
                                try:
                                    # Get PDF URL and filename
                                    pdf_url = pdf_link.get_attribute("href")
                                    filename = pdf_url.split("/")[-1]

                                    # skip non-PDF files
                                    if not filename.lower().endswith(".pdf"):
                                        continue

                                    # Check if file already exists in database
                                    docInDB = find_document_by_url(
                                        mysql_driver, pdf_url
                                    )
                                    if docInDB:
                                        continue

                                    # Create document object
                                    document = Document(
                                        number=filename.split(".")[0],
                                        title=page_title,
                                        description="",
                                        url=pdf_url,
                                        documentType=scrapeScript.documentTypeID,
                                        documentStatus=1,  # Active
                                        activeDate=datetime.today().date(),
                                        inactiveDate=None,
                                        sourceFileName=None,
                                        pdfFileName=None,
                                        parsed=False,
                                        embedded=False,
                                        parsingScriptID=scrapeScript.defaultParsingScriptID,
                                        createdDate=datetime.today().date(),
                                        modifiedDate=datetime.today().date(),
                                        fileSize=0,
                                        parsingLog="notParsed",
                                        embeddingLog="notEmbedded",
                                        noOfParagraphs=0,
                                        lastScrapeDate=datetime.today().date(),
                                        sourceProject=0,
                                    )

                                    # Download the file
                                    doc, msgText = download_file(
                                        config, document, len(download_list) + 1
                                    )
                                    if doc is not None:
                                        download_list.append(doc)
                                    else:
                                        logList.append(msgText)
                                        scrape_url_append_log(
                                            mysql_driver, scrapeURLId, msgText
                                        )

                                except Exception as e:
                                    print(f"Error processing PDF link: {str(e)}")
                                    continue

                        except TimeoutException:
                            print(
                                f"Timeout waiting for PDF links in tab {window_handle}"
                            )
                            continue

                    except Exception as e:
                        print(f"Error processing tab: {str(e)}")
                    finally:
                        # Close the current tab
                        driver.close()

                # Switch back to main window
                driver.switch_to.window(main_window)

            # Try to click next page button
            try:
                next_button = wait.until(
                    EC.presence_of_element_located(
                        (By.XPATH, "//a[contains(text(), 'next')]")
                    )
                )
                if not next_button.is_enabled():
                    print("No more pages to process")
                    break
                next_button.click()
                page_num += 1

                # Wait for the next page to load
                wait.until(
                    EC.presence_of_all_elements_located(
                        (
                            By.XPATH,
                            "//a[contains(@href, '/pubs/') and not(contains(@href, 'ipd'))]",
                        )
                    )
                )
            except TimeoutException:
                print("Timeout waiting for next page to load")
                break
            except Exception as e:
                print(f"Error navigating to next page: {str(e)}")
                break

        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            break

    return download_list


def run(config: ScriptsConfig, scrapeURLId):
    global logList
    datetime.today().date()
    driver = None

    mysql_driver = MySQLDriver(cred=config.databaseConfig.__dict__)

    logText = f"Scrape for NIST Publications started at {datetime.today()}"
    logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    print(logText)

    try:
        driver = get_main_page(config)
        # Search for FIPS and SP publications
        search_series(driver, ["SP", "FIPS"])
        download_list = process_page(config, driver, mysql_driver, scrapeURLId)

        # Insert downloaded documents into database
        if download_list and len(download_list) > 0:
            docIDsList = insert_documents_bulk2(mysql_driver, download_list)
            if docIDsList is not None and len(docIDsList) > 0:
                print(f"Newly added docIDs: {docIDsList}")
                logText = f"Newly added docIDs: {docIDsList}"
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)

    except Exception:
        logText = traceback.format_exc()
        logList.append(logText)
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)
        print(logText)
    finally:
        if driver is not None:
            driver.quit()
        mysql_driver.close()

    logText = f"Scraping NIST Publications DONE"
    logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    print(logText)


if __name__ == "__main__":
    try:

        props = None
        docIdsList = []
        if len(sys.argv) > 1:
            n = len(sys.argv[1])
            docs = sys.argv[1][1 : n - 1]
            docs = docs.split(" ")
            docIdsList = [int(i) for i in docs]

        if len(docIdsList) > 0:
            scrapeURLId = docIdsList[0]
        else:
            scrapeURLId = 2

        configs = parseCredentialFile(str(CONFIG_DIR / "dev_test_tlp_config.json"))

        if configs:
            run(configs, scrapeURLId)

    except Exception as e:
        print("The EXCEPTION >>>>>>>>>>>>>> ")
        print(traceback.format_exc())
        traceback.print_exc()
        print(e)
