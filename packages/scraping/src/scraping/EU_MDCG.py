import requests
import regex as re
from typing import Optional, Tuple, Union

from bs4 import BeautifulSoup
import csv
import time
import json
import logging
import os
import re
from urllib.parse import urljoin
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("mdcg_scraper.log", mode='w'),
        logging.StreamHandler()
    ]
)

# Folder to store downloaded files
DOWNLOAD_DIR = Path("scrapedDocs")
DOWNLOAD_DIR.mkdir(exist_ok=True)

def sanitize_filename(name):
    """Sanitize filename by removing problematic characters"""
    return re.sub(r'[^\w\-_\. ]', '_', name)

def validate_item(item):
    """Check that required fields are present"""
    return bool(item['reference_number'] and item['title'] and item['publication_date'])


def extract_revision_info(text: str) -> Optional[Tuple[str, Union[int, list]]]:
    """
    Extracts revision info (e.g., 'rev.1', 'rev.1.2') from the input string.

    Returns:
        A tuple like ('rev', 1) or ('rev', [1, 2]) or None if not found.
    """
    match = re.search(r'\brev\.([0-9]+(?:\.[0-9]+)*)', text, re.IGNORECASE)
    if match:
        version_str = match.group(1)
        parts = version_str.split('.')
        numbers = [int(p) for p in parts]
        if len(numbers) == 1:
            return ('rev', numbers[0])
        else:
            return ('rev', numbers)
    return None


def fetch_guidance_list(url):
    logging.info(f"Fetching guidance list from: {url}")
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    base_url = resp.url
    guidance_items = []

    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if not {"Reference", "Title", "Publication"}.issubset(set(headers)):
            continue

        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) < 3:
                continue

            ref = cols[0].get_text(" ", strip=True)
            title = cols[1].get_text(" ", strip=True)
            pub = cols[2].get_text(" ", strip=True)

            links = {"pdf": None, "docx": None}
            for cell in cols:
                for a in cell.find_all("a", href=True):
                    href = a["href"].strip()
                    full_url = urljoin(base_url, href)
                    if href.lower().endswith(".pdf"):
                        links["pdf"] = links["pdf"] or full_url
                    elif href.lower().endswith((".docx", ".doc")):
                        links["docx"] = links["docx"] or full_url

            item = {
                "reference_number": ref,
                "title": title,
                "publication_date": pub,
                "links": links
            }

            if validate_item(item):
                guidance_items.append(item)
            else:
                logging.warning(f"Skipping invalid item: {item}")

    logging.info(f"Found {len(guidance_items)} valid guidance entries")
    return guidance_items

def download_file(url, reference_number, filetype):
    """Download the file to scrapedDocs/ folder if not already downloaded"""
    try:
        filename = sanitize_filename(f"{reference_number}_{filetype}.{url.split('.')[-1]}")
        filepath = DOWNLOAD_DIR / filename

        if filepath.exists():
            logging.info(f"File already exists: {filepath}")
            return str(filepath)

        logging.info(f"Downloading {filetype.upper()} from {url}")
        response = requests.get(url, stream=True, timeout=20)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"Saved: {filepath}")
        return str(filepath)

    except Exception as e:
        logging.error(f"Failed to download {filetype.upper()} from {url}: {e}")
        return None

def download_all_documents(data):
    count = 0
    for item in data:
        
        ref = item["reference_number"]
        for filetype in ["pdf", "docx"]:

            url = item["links"].get(filetype)
            if url:
                count += 1
                if count % 5:
                    time.sleep(5)
                local_path = download_file(url, ref, filetype)
                item["links"][filetype] = local_path  # update with local path if downloaded

def save_files(data):
    try:
        with open("mdcg_guidance.json", "w", encoding="utf-8") as f_json:
            json.dump(data, f_json, ensure_ascii=False, indent=2)
        logging.info("Saved JSON file: mdcg_guidance.json")

        with open("mdcg_guidance.csv", "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=["reference_number", "title", "publication_date", "pdf_link", "docx_link"])
            writer.writeheader()
            for item in data:
                writer.writerow({
                    "reference_number": item["reference_number"],
                    "title": item["title"],
                    "publication_date": item["publication_date"],
                    "pdf_link": item["links"].get("pdf"),
                    "docx_link": item["links"].get("docx")
                })
        logging.info("Saved CSV file: mdcg_guidance.csv")

    except Exception as e:
        logging.error(f"Error saving output files: {e}")

def main():
    logging.info("🔍 Starting MDCG scraper...")
    url = "https://health.ec.europa.eu/medical-devices-sector/new-regulations/guidance-mdcg-endorsed-documents-and-other-guidance_en#sec3"
    try:
        data = fetch_guidance_list(url)
        download_all_documents(data)
        save_files(data)
    except Exception as e:
        logging.error(f"Unhandled error: {e}")
    logging.info("✅ Scraping complete.")

if __name__ == "__main__":
    main()
