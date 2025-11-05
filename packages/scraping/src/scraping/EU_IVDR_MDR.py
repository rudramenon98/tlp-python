import os
import time
from urllib.parse import urljoin

import requests
from database.utils.WebDriverFactory import WebDriverFactory
from selenium.common.exceptions import TimeoutException

# from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def setup_driver(docker_url):
    driver = WebDriverFactory.getWebDriverInstance(
        browser="docker", docker_url=docker_url
    )
    return driver


def download_file(url, filename, download_dir):
    os.makedirs(download_dir, exist_ok=True)
    # filename = os.path.basename(url.split('?')[0])
    filepath = os.path.join(download_dir, filename)

    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return filepath

    print(f"Downloading: {url}")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def wait_for_page_load(driver, timeout=15):
    try:
        # Wait for at least one <a> tag to appear (or a specific container class if known)
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.TAG_NAME, "a"))
        )
        print("✅ Page fully loaded.")
    except TimeoutException:
        print("⚠️ Timeout waiting for page to load.")


def scrape_consolidated_documents():
    base_url = "https://eur-lex.europa.eu/eli/reg/2017/746/oj/eng"
    download_dir = "downloads"

    driver = setup_driver(docker_url="http://localhost:4444")
    driver.get(base_url)

    wait_for_page_load(driver)
    # Wait and find the "Current consolidated version" link
    time.sleep(5)
    """
    try:
        consolidated_link = driver.find_element(By.PARTIAL_LINK_TEXT, "Current consolidated version")
        consolidated_url = consolidated_link.get_attribute("href")
        print(f"Found consolidated version link: {consolidated_url}")
    except Exception as e:
        driver.quit()
        raise RuntimeError("Could not find the 'Current consolidated version' link") from e
        #PP1Contents > div > p.forceIndicator > strong > span > a
    """

    # Find all <a> tags and look for "Current consolidated version" in text
    consolidated_url = None
    """
    links = driver.find_elements(By.TAG_NAME, "a")
    for link in links:
        text = link.text.strip().lower()
        print(text)
        if "current consolidated version" in text.lower():
            consolidated_url = link.get_attribute("href")
            break
    """
    link = driver.find_element(
        By.CSS_SELECTOR, "#PP1Contents > div > p.forceIndicator > strong > span > a"
    )
    consolidated_url = link.get_attribute("href")
    if not consolidated_url:
        driver.quit()
        raise RuntimeError(
            "Could not find the 'Current consolidated version' link using <a> tags"
        )

    print(f"✅ Found consolidated version link: {consolidated_url}")

    # Navigate to the consolidated version page
    driver.get(consolidated_url)
    wait_for_page_load(driver)
    time.sleep(5)

    # Look for all language-format links (PDF, HTML, etc.)
    links = driver.find_elements(By.CSS_SELECTOR, "a[href]")
    en_pdf = None
    en_html = None

    for link in links:
        title = link.get_attribute("title") or ""
        href = link.get_attribute("href") or ""
        title_lower = title.lower()

        if "english" in title_lower and "pdf" in title_lower:
            en_pdf = urljoin(driver.current_url, href)
        elif "english" in title_lower and "html" in title_lower:
            en_html = urljoin(driver.current_url, href)

    driver.quit()

    print("\n--- Found Links ---")
    print("EN HTML:", en_html or "Not found")
    print("EN PDF:", en_pdf or "Not found")

    # Download the documents
    if en_html:
        download_file(en_html, "ivdr_mdf_regulations_en.html", download_dir)
    if en_pdf:
        download_file(en_pdf, "ivdr_mdf_regulations_en.pdf", download_dir)


if __name__ == "__main__":
    scrape_consolidated_documents()
