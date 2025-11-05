import ftplib
import gzip
import hashlib
import logging
import os
import random
import time
import traceback
from datetime import datetime
from math import ceil
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Union

import regex as re
import requests
from common_tools.log_config import configure_logging_from_argv

# Replace with your actual imports or implementations
from database.document_service import (
    find_documents_by_type,
    get_scrape_script_by_scraperUrlId,
    get_scrapeurl_by_scrapeUrlId,
    insert_document,
    update_documents,
)
from database.entity.Document import Document
from database.entity.PubMed import (  # adjust import to your project structure
    Base,
    PubMed,
)
from database.entity.Repository import Repository
from database.entity.ScrapScript import ScrapScript
from database.entity.ScriptsProperty import parseCredentialFile
from database.utils.MySQLFactory import MySQLDriver
from lxml import html
from pubmed_parser import PubmedFullParser

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DateToday = datetime.today().date()
# ------------------ Exception ------------------


class DBCredentialError(Exception):
    pass


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],  # You can add FileHandler here if needed
)

log = logging.getLogger(__name__)


def extract_pubmed_download_date(url: str, xpath: str) -> datetime.date:
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        log.error(f"Failed to fetch {url}: {e}")
        raise

    tree = html.fromstring(response.content)

    elements = tree.xpath(xpath)
    if not elements:
        log.error(f"No content found at XPath: {xpath}")
        raise ValueError("No matching content at XPath")

    text = elements[0].text_content().strip()
    log.debug(f"Extracted text: {text}")

    # Look for a date string
    match = re.search(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
        text,
    )
    if not match:
        log.error("No valid date found in the extracted text.")
        raise ValueError("Date not found")

    date_str = match.group(0)
    date_obj = datetime.strptime(date_str, "%B %d, %Y").date()
    log.debug(f"Extracted date: {date_obj}")
    return date_obj


# ------------------ FTP & File Functions ------------------


def download_file(ftp, filename, local_path):
    with open(local_path, "wb") as f:
        ftp.retrbinary(f"RETR {filename}", f.write)


def calculate_md5(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def verify_md5(local_file, md5_file):
    with open(md5_file, "r") as f:
        md5_expected = f.read().split()[-1]
    md5_actual = calculate_md5(local_file)
    return md5_actual == md5_expected


def check_file_exists_and_size_same(filepath, remote_size):
    """
    Checks if the local file exists and compares its size in bytes with remote size.

    Parameters:
        filepath (str): The path to the file.

    Returns:
        exists (bool)
    """
    if os.path.isfile(filepath):
        local_size = os.path.getsize(filepath)
        if local_size != remote_size:
            print(f"Size mismatch: local={local_size}, remote={remote_size}")
            return False
        else:
            print(f"Sizes match: local={local_size}, remote={remote_size}")
            return True
    else:
        return False


def check_local_file_exists(filepath):
    """
    Checks if the local file exists.

    Parameters:
        filepath (str): The path to the file.

    Returns:
        exists (bool)
    """
    if os.path.isfile(filepath):
        local_size = os.path.getsize(filepath)
        if local_size > 0:
            print(f"File already downloaded earlier = {filepath}")
            return True
        else:
            print(f"File NOT already downloaded = {filepath}")
            return False
    else:
        return False


def download_files(
    ftp_server: str, ftp_folder: str, download_folder: str, update: bool = False
):

    downloaded_files = []

    os.makedirs(download_folder, exist_ok=True)

    def connect():
        ftp = ftplib.FTP(ftp_server)
        ftp.login()
        ftp.cwd(ftp_folder)
        ftp.voidcmd("TYPE I")
        return ftp

    # Initial connection to fetch file list
    ftp = connect()
    files = ftp.nlst()
    ftp.quit()

    gz_files = [f for f in files if f.endswith(".gz")]

    if update:
        # Get set of local files
        local_files = set(os.listdir(download_folder))
        # Filter out files that are already present locally
        gz_files = [f for f in gz_files if f not in local_files]
        logging.info(f"Update mode: {len(gz_files)} new files to download.")

    total_files = len(gz_files)
    batch_size = 100

    logging.info(f"Found {total_files} .gz files to download.")
    for i in range(0, total_files, batch_size):
        batch = gz_files[i : i + batch_size]

        ftp = connect()
        logging.info(f"Started batch {(i // batch_size) + 1} with {len(batch)} files.")

        for gz_file in batch:
            md5_file = gz_file + ".md5"
            local_gz_path = os.path.join(download_folder, gz_file)
            local_md5_path = os.path.join(download_folder, md5_file)

            # Always download the .md5 file first (we need it to verify either way)
            logging.info(f"Downloading {md5_file} for verification...")
            download_file(ftp, md5_file, local_md5_path)
            time.sleep(random.uniform(0.25, 0.85))

            needs_download = True

            if check_local_file_exists(local_gz_path):
                logging.info(f"Local file exists: {local_gz_path}")

                if verify_md5(local_gz_path, local_md5_path):
                    logging.info(f"{gz_file}: MD5 verified, skipping download.")
                    needs_download = False
                else:
                    logging.warning(f"{gz_file}: MD5 mismatch, re-downloading...")

            if needs_download:
                # Only download the .gz file now
                logging.info(f"Downloading {gz_file}...")
                download_file(ftp, gz_file, local_gz_path)
                time.sleep(random.uniform(0.35, 1.57))

                logging.info(f"Verifying {gz_file} integrity after download...")

                if verify_md5(local_gz_path, local_md5_path):
                    logging.info(f"{gz_file}: OK")
                    downloaded_files.append(gz_file)
                else:
                    logging.warning(f"{gz_file}: FAILED checksum verification")
        ftp.quit()
        logging.info(f"Completed batch {(i // batch_size) + 1}.")

        # Sleep only if there are more files to process
        if i + batch_size < total_files:
            sleep_time = random.randint(10, 120)
            logging.info(f"Sleeping for {sleep_time} seconds before next batch...")
            time.sleep(sleep_time)

    logging.info("All files downloaded.")

    return downloaded_files


def delete_obsolete_local_files(folder_path: str) -> int:
    """
    Deletes all *.gz and *.md5 files in the given folder (non-recursively).

    Args:
        folder_path (str): Path to the target folder.

    Returns:
        int: Number of files deleted.
    """
    deleted_count = 0
    folder = Path(folder_path)

    if not folder.is_dir():
        log.error(f"Provided path is not a directory: {folder_path}")
        return 0

    for ext in ("*.gz", "*.md5"):
        for file in folder.glob(ext):  # change to rglob for include subfolders
            try:
                file.unlink()
                deleted_count += 1
                log.info(f"Deleted: {file}")
            except Exception as e:
                log.warning(f"Failed to delete {file}: {e}")

    log.info(f"Total files deleted: {deleted_count}")
    return deleted_count


# Replace with your actual parser import
# from pubmed_parser import PubmedFullParser

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def text_normalize(key: str) -> str:
    """Normalize the key by removing whitespace and converting to lowercase."""
    return "".join(key.split()).lower()


#        return key.lower()


# Batch function: each process handles a list of files
def process_batch(args):
    db_cred, file_paths, document_id = args

    # Create a separate DB connection in each process
    mysql_driver = MySQLDriver(cred=db_cred)

    for file_path in file_paths:
        parse_and_upsert_pubmed_file(mysql_driver, file_path, document_id)


def chunk_list(lst, chunk_size):
    """Helper to split a list into chunks"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def parse_and_upsert_pubmed_file(
    mysql_driver, file_path: Union[str, os.PathLike], documentID: int
) -> bool:
    """
    Parses a compressed PubMed XML update file and upserts articles into the database.

    Args:
        file_path (str): Path to the .xml.gz file.
        db_uri (str): SQLAlchemy database URI (e.g., mysql+mysqlconnector://user:pass@host/db)

    Returns:
        bool: True if success, False otherwise.
    """
    try:
        if not os.path.isfile(file_path) or not file_path.endswith(".xml.gz"):
            log.error(f"Invalid file path: {file_path}")
            return False

        session = mysql_driver.get_session()
        with gzip.open(file_path, "rb") as f:
            parser = PubmedFullParser(f)
            count = 0
            for art in parser.parse():
                count += 1

                try:
                    pmid = int(art.pmid)
                except Exception:
                    log.warning(
                        f"pmid: {art.pmid} cannot be converted to int, skipping"
                    )
                    continue

                try:
                    pmidversion = int(getattr(art, "pmid_version", 1))
                except Exception:
                    log.debug(
                        f"pmidversion: {art.pmid_version} cannot be converted to int, set to 1 "
                    )
                    pmidversion = 1

                is_deleted = bool(getattr(art, "deleted", False))
                title = getattr(art, "title", None)
                if title:
                    title = title[:2048]

                abstract = getattr(art, "abstract", None)
                if abstract:
                    abstract = abstract[:8192]

                if art.authors:
                    authors = "; ".join(
                        f"{a.last_name}, {a.fore_name}" for a in art.authors
                    )
                    authors = authors[:2048]
                else:
                    authors = ""

                created = getattr(art, "created_date", None)
                modified = getattr(art, "modified_date", None)

                # if not (art.abstract and len(art.abstract) > 0 and art.doi and len(art.doi) > 0):
                #    continue

                if is_deleted:
                    log.debug(f"[DELETED] PMID {pmid}, Title: {title}")
                #                elif not abstract : #or not art.doi:
                #                    log.debug(f"[] PMID {pmid}: missing abstract.")
                else:
                    log.debug(f"[UPSERT] PMID {pmid} v{pmidversion}: {title}")

                # Check if record exists
                existing = session.query(PubMed).filter(PubMed.pmid == pmid).first()

                if existing:
                    if is_deleted:
                        log.debug(f"[DELETED] PMID {pmid}, Title: {title}")
                        existing.deleted = is_deleted
                        existing.modifiedDate = DateToday
                        session.flush()
                    elif existing.pmidversion < pmidversion:
                        # Update Repository records for title and abstract if the texts have changed

                        if title and text_normalize(existing.title) != text_normalize(
                            title
                        ):
                            currTitleParagraph = (
                                session.query(Repository)
                                .filter(
                                    Repository.paragraphID == existing.titleParagraphID
                                )
                                .first()
                            )
                            if currTitleParagraph:
                                currTitleParagraph.data = title

                        if (
                            abstract
                            and existing.abstract
                            and text_normalize(existing.abstract)
                            != text_normalize(abstract)
                        ):
                            currAbstractParagraph = (
                                session.query(Repository)
                                .filter(
                                    Repository.paragraphID
                                    == existing.abstractParagraphID
                                )
                                .first()
                            )
                            if currAbstractParagraph:
                                currAbstractParagraph.data = abstract

                        # Update existing record as new version has been scraped
                        existing.pmidversion = pmidversion
                        existing.title = title[:1024]
                        existing.abstract = abstract[:8192]
                        existing.deleted = is_deleted
                        existing.authors = authors[:2048]
                        existing.createdDate = created
                        existing.modifiedDate = modified or DateToday
                        session.flush()
                        log.debug(f"[UPDATED] PMID {pmid}, Title: {title}")
                    else:
                        pass
                else:
                    titleParagraphID = 0
                    abstractParagraphID = 0

                    # First insert title and abstract into two Repository records
                    if title and len(title) > 0:
                        titleParagraph = Repository(
                            documentID=documentID,
                            data=title,
                            Type=2,  # Heading for title
                            pageNo=0,
                            wordCount=len(title[:1024]),
                        )
                        session.add(titleParagraph)
                        session.flush()
                        titleParagraphID = titleParagraph.paragraphID

                    if abstract and len(abstract) > 0:
                        abstractParagraph = Repository(
                            documentID=documentID,
                            data=abstract[:8192],
                            Type=0,  # paragraph for abstract
                            pageNo=0,
                            wordCount=len(abstract[:8192]),
                        )

                        session.add(abstractParagraph)
                        session.flush()
                        abstractParagraphID = abstractParagraph.paragraphID

                    log.debug("titleParagraphID (after flush):", titleParagraphID)
                    log.debug("abstractParagraphID:", abstractParagraphID)

                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}.{pmidversion}"
                    # Insert new
                    new_article = PubMed(
                        pmid=pmid,
                        pmidversion=pmidversion,
                        title=title,
                        abstract=abstract or "",
                        deleted=is_deleted,
                        authors=authors,
                        url=url,
                        titleParagraphID=titleParagraphID,
                        abstractParagraphID=abstractParagraphID,
                        createdDate=created or DateToday,
                        modifiedDate=modified or DateToday,
                    )
                    session.add(new_article)
                    log.debug(f"[INSERTED] PMID {pmid}, Title: {title}")

                if count % 100 == 0:
                    session.commit()
                    log.debug(f"Processed {count} articles...")

            session.commit()
            log.info(
                f"✅ Processed: {file_path} ==> Parsed and upserted {count} articles."
            )

        # session.close()
        return True
    except Exception:
        log.exception(
            f"❌ Processing: {file_path} ==> Failed to parse and insert PubMed data:"
        )
        if session:
            session.rollback()
        return False
    finally:
        if session:
            session.close()
        return True


def get_pubmed_document(mysql_driver, scrapeURLId: int):
    scrapeScript: ScrapScript = get_scrape_script_by_scraperUrlId(
        mysql_driver, scrapeURLId
    )

    doc = find_documents_by_type(mysql_driver, scrapeScript.documentTypeID)

    if not doc:
        document = Document(
            number="PubMed",
            title="PubMed",
            description="PubMed Dummy Document",
            url="https://pubmed.ncbi.nlm.nih.gov/",
            documentType=scrapeScript.documentTypeID,
            documentStatus=1,  # Active
            activeDate=datetime.today().date(),
            inactiveDate=None,
            sourceFileName=None,
            pdfFileName=None,
            parsed=True,
            embedded=False,
            parsingScriptID=scrapeScript.defaultParsingScriptID,
            createdDate=datetime.today().date(),
            modifiedDate=datetime.today().date(),
            fileSize=0,
            parsingLog="Parsed",
            embeddingLog="notEmbedded",
            noOfParagraphs=0,
            lastScrapeDate=datetime.today().date(),
            sourceProject=0,
        )

        # insert document in DB
        insert_document(mysql_driver, document)

        doc = find_documents_by_type(mysql_driver, scrapeScript.documentTypeID)
        log.info(f"inserted PubMed document => DocumentId: {doc.documentId}")

    return doc


# ------------------ Main ------------------


def main():
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
            scrapeURLId = 3

        config_path = "tlp-scripts/test-tlp_config.json"
        config = parseCredentialFile(config_path)
        if config is None:
            LOCAL_DIR = "./pubmed/baseline_files"
            LOCAL_UPDATE_DIR = "./pubmed/daily_updates"
        else:
            LOCAL_DIR = config.baseDir + "/scrapedDocs/pubmed/baseline_files"
            LOCAL_UPDATE_DIR = config.baseDir + "/scrapedDocs/pubmed/daily_updates"

        url = "https://pubmed.ncbi.nlm.nih.gov/download/"
        xpath = "/html/body/main/div[2]/p[1]"

        pubmed_release_date = extract_pubmed_download_date(url, xpath)
        today = datetime.today().date()

        log.info(f"PubMed data last updated on: {pubmed_release_date}")

        # establish database connection
        dbCredentials = config.databaseConfig.__dict__
        mysql_driver = MySQLDriver(cred=dbCredentials)

        baseline_stale = False

        # Get the scrape URL object to obtain the last scraped date
        scrapeURL_data = get_scrapeurl_by_scrapeUrlId(mysql_driver, scrapeURLId)
        if scrapeURL_data:
            lastScrapeDate = scrapeURL_data.lastScrapeDate
        else:
            lastScrapeDate = None

        if lastScrapeDate:
            days_ago = 0

            if pubmed_release_date <= today:
                days_ago = (pubmed_release_date - lastScrapeDate).days
                if days_ago >= 0:
                    baseline_stale = True
                else:
                    baseline_stale = False
            else:
                baseline_stale = False
        else:
            baseline_stale = True

        # set the FTP server details
        FTP_HOST = "ftp.ncbi.nlm.nih.gov"
        FTP_BASELINE_DIR = "/pubmed/baseline"
        FTP_UPDATE_DIR = "/pubmed/updatefiles"

        pubmed_doc = get_pubmed_document(mysql_driver, scrapeURLId)

        if baseline_stale:
            # remove previous local files
            # delete_obsolete_local_files(LOCAL_DIR)
            # delete_obsolete_local_files(LOCAL_UPDATE_DIR)
            downloaded_files = download_files(
                FTP_HOST, FTP_BASELINE_DIR, LOCAL_DIR, update=False
            )

            """
            files = [f for f in os.listdir(LOCAL_DIR) if f.endswith('.gz')]

            for file in files:
                file_path = os.path.join(LOCAL_DIR, file)
                parse_and_upsert_pubmed_file(mysql_driver, file_path, pubmed_doc.documentId)
            """

            if downloaded_files and len(downloaded_files) > 0:
                # Find all .gz files
                # files = [f for f in os.listdir(LOCAL_DIR) if f.endswith('.gz') and f in downloaded_files]
                files = [f for f in downloaded_files]
                file_paths = [os.path.join(LOCAL_DIR, f) for f in files]

                # Split files into batches
                num_processes = max(1, cpu_count() // 4)
                batch_size = ceil(len(file_paths) / num_processes)

                if batch_size > 0:
                    batches = list(chunk_list(file_paths, batch_size))

                    # Prepare args for each batch

                    args_list = [
                        (dbCredentials, batch, pubmed_doc.documentId)
                        for batch in batches
                    ]

                    # Use multiprocessing
                    with Pool(processes=num_processes) as pool:
                        pool.map(process_batch, args_list)

            # when new baseline is published, new update files need to used!
            # delete_obsolete_local_files(LOCAL_UPDATE_DIR)

        downloaded_files = download_files(
            FTP_HOST, FTP_UPDATE_DIR, LOCAL_UPDATE_DIR, update=True
        )
        if downloaded_files and len(downloaded_files) > 0:
            """
            files = [f for f in os.listdir(LOCAL_UPDATE_DIR) if f.endswith('.gz')]

            for file in files:
                file_path = os.path.join(LOCAL_UPDATE_DIR, file)
                parse_and_upsert_pubmed_file(mysql_driver, file_path, pubmed_doc.documentId)
            """
            # Find all .gz files
            # files = [f for f in os.listdir(LOCAL_UPDATE_DIR) if f.endswith('.gz') and f in downloaded_files]
            files = [f for f in downloaded_files]
            file_paths = [os.path.join(LOCAL_UPDATE_DIR, f) for f in files]

            # Split files into batches
            num_processes = max(1, cpu_count() // 4)

            batch_size = ceil(len(file_paths) / num_processes)

            if batch_size > 0:
                batches = list(chunk_list(file_paths, batch_size))

                # Prepare args for each batch

                args_list = [
                    (dbCredentials, batch, pubmed_doc.documentId) for batch in batches
                ]

                # Use multiprocessing
                with Pool(processes=num_processes) as pool:
                    pool.map(process_batch, args_list)

        # update the lastScrapeDate in the document
        pubmed_doc.lastScrapeDate = datetime.today().date()
        update_documents(mysql_driver, [pubmed_doc])

        # process_update_files(session)
        log.info(f"✅ All done: PubMed database scraped on {datetime.today().date()}!")

    except Exception as e:
        log.error("The EXCEPTION >>>>>>>>>>>>>> ")
        log.error(traceback.format_exc())
        # traceback.print_exc()
        log.exception(e)


if __name__ == "__main__":
    main()
