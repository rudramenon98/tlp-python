## Library import
import io
import json
import logging
import os
import traceback
from datetime import date, datetime

import pandas as pd
import requests
from app.document_service import (
    cancel_documents,
    find_document_by_url,
    get_scrape_script_by_scraperUrlId,
    insert_document,
    update_documents,
)
from app.entity.Document import StaticPublicDocument
from app.entity.ScrapScript import ScrapScript
from app.entity.ScriptsProperty import ScriptsConfig, parseCredentialFile
from app.scrape_url_service import (
    scrape_url_append_log,
    update_scrape_url_set_log_value,
)
from app.utils.MySQLFactory import MySQLDriver
from app.utils.util import get_dir_safe
from logconfig import configure_logging_from_argv
from PyPDF2 import PdfFileReader, PdfReader
from tqdm import tqdm

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
'''
# Console (stdout) handler
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
'''

# for logging
logList = []


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


##Checking availabllity if document exists or not
def check_if_file_exists2(link):
    # link = 'https://www.govinfo.gov/content/pkg/CFR-'+str(year)+'-title'+str(title)+'-vol'+str(vol)+'/pdf/CFR-'+str(year)+'-title'+str(title)+'-vol'+str(vol)+'.pdf'
    # print(link)
    hdr = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9,gu;q=0.8,hi;q=0.7",
        "Connection": "keep-alive",
    }

    page = requests.get(str(link), headers=hdr)
    with io.BytesIO(page.content) as f:
        try:
            PdfFileReader(f)
            # print(pdf)
            return True
        except:
            # print('Year',year,'Title',title,'Vol',vol,'is not found')
            return False


def check_if_file_exists3(link):
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
            log.debug(
                "Warning: %s downloaded %s out of %s bytes",
                str(link),
                str(size),
                str(total),
            )
        try:
            PdfReader(stream)
            # print(pdf)
            return True
        except:
            # print('Year',year,'Title',title,'Vol',vol,'is not found')
            return False

    return False


##Checking availabllity if document exists or not
def check_availability(year, title, vol, type):
    link = (
        "https://www.govinfo.gov/content/pkg/CFR-"
        + str(year)
        + "-title"
        + str(title)
        + "-vol"
        + str(vol)
        + "/"
        + type
        + "/CFR-"
        + str(year)
        + "-title"
        + str(title)
        + "-vol"
        + str(vol)
        + "-chapI-subchapH."
        + type
    )
    log.debug("Checking availability for link: %s", link)

    file_exists = check_if_file_exists3(link)

    if file_exists:
        log.debug("Year %s Title %s Vol %s is found", year, title, vol)
        return True
    else:
        log.debug("Year %s Title %s Vol %s is not found", year, title, vol)
        return False


def get_cfr_path(title, year, vol, type):
    cfr_path = (
        "https://www.govinfo.gov/content/pkg/CFR-"
        + str(year)
        + "-title"
        + str(title)
        + "-vol"
        + str(vol)
        + "/"
        + type
        + "/CFR-"
        + str(year)
        + "-title"
        + str(title)
        + "-vol"
        + str(vol)
        + "-chapI-subchapH."
        + type
    )
    return cfr_path


def get_fda_cfrs():
    global logList
    ID = []
    Title = []
    Description = []
    PDF_FileName = []
    PDF_FileUrl = []
    XML_FILEName = []
    XML_FileUrl = []
    FileType = []
    Active_date = []
    Curr_Year = []

    ##Getting Current Year
    today = datetime.today().date()
    year = today.year
    print("This Year :", year)

    # there are 5 volumes in FAA CFRs
    volumes = [8]
    number_of_volums = 1

    # the FAA CFRs are in title14
    number_of_titles = [21]

    for i in range(1, number_of_volums + 1):
        for j in number_of_titles:
            current_year = year
            idx = 0
            while current_year != 1996:

                if check_availability(
                    current_year, j, volumes[i - 1], "pdf"
                ):  # and check_availability(current_year, j, i, 'xml'):

                    idx += 1

                    TITLE = (
                        "CFR-"
                        + str(current_year)
                        + "-title"
                        + str(j)
                        + "-vol"
                        + str(volumes[i - 1])
                        + "-chapI-subchapH."
                    )
                    # pdf_link = 'https://www.govinfo.gov/content/pkg/CFR-'+str(current_year)+'-title'+str(j)+'-vol'+str(i)+'/pdf/CFR-'+str(current_year)+'-title'+str(j)+'-vol'+str(i)+'.pdf'
                    # xml_link = 'https://www.govinfo.gov/content/pkg/CFR-'+str(current_year)+'-title'+str(j)+'-vol'+str(i)+'/xml/CFR-'+str(current_year)+'-title'+str(j)+'-vol'+str(i)+'.xml'

                    pdf_link = get_cfr_path(j, current_year, volumes[i - 1], "pdf")
                    print("PDF link:", pdf_link)

                    xml_link = get_cfr_path(j, current_year, volumes[i - 1], "xml")
                    print("XML link:", xml_link)

                    print("TITLE", TITLE)

                    xml_file_name = TITLE + ".xml"
                    pd_file_name = TITLE + ".pdf"

                    file_type = 1  # 1 CFR

                    date = "01/01/" + str(current_year)

                    # store the values
                    ID.append(idx)
                    Title.append(TITLE)
                    Description.append(TITLE.replace("-", " "))
                    PDF_FileName.append(pd_file_name)
                    PDF_FileUrl.append(pdf_link)
                    XML_FILEName.append(xml_file_name)
                    XML_FileUrl.append(xml_link)
                    FileType.append(file_type)
                    Active_date.append(date)
                    Curr_Year.append(current_year)
                    break

                current_year = current_year - 1

    # create DF
    df = pd.DataFrame()
    df["ID"] = [i for i in range(len(Title))]

    # store values in DF
    df["title"] = Title
    df["Number"] = Title
    df["description"] = Description
    df["pdf_filename"] = PDF_FileName
    df["pdf_file_url"] = PDF_FileUrl
    df["xml_filename"] = XML_FILEName
    df["xml_file_url"] = XML_FileUrl
    df["filetype"] = FileType
    df["active_date"] = Active_date
    df["curr_year"] = Curr_Year

    # for debugging
    # df.to_csv('./US_CFR_Scrapping_04102023.csv', index=False)

    return df


def download_file(URL, path):
    hdr = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9,gu;q=0.8,hi;q=0.7",
        "Connection": "keep-alive",
    }
    chunk_size = (1024 * 1024) * 1  # 1MB

    response = requests.get(URL, headers=hdr, stream=True)
    total = int(response.headers.get("content-length", 0))

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


def download_cfr_document(
    config: ScriptsConfig, mysql_driver: MySQLDriver, df, scrapeURLId
):

    global logList
    try:
        XML_URL = df["xml_file_url"]
        PDF_URL = df["pdf_file_url"]

        # check if document exists in documents table
        docInDB = find_document_by_url(mysql_driver, PDF_URL, doc_class=StaticPublicDocument)

        if not docInDB:

            file_name = XML_URL.split("/")[-1]
            pdf_file_name = PDF_URL.split("/")[-1]

            # folder = config.rootDataDir + config.downloadDirInRootDir
            folder = config.downloadDir
            path = get_dir_safe(folder) + "/" + file_name
            pdf_path = get_dir_safe(folder) + "/" + pdf_file_name

            try:
                download_file(XML_URL, path)
            except Exception:
                logText = (
                    df["xml_filename"]
                    + " with filetype = "
                    + str(df["filetype"])
                    + " at "
                    + df["xml_file_url"]
                    + " download failed on "
                    + str(datetime.today().date())
                )
                logText += "\n " + traceback.format_exc()
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                return False

            try:
                download_file(PDF_URL, pdf_path)
            except Exception:
                logText = (
                    df["pdf_filename"]
                    + " with filetype = "
                    + str(df["filetype"])
                    + " at "
                    + df["pdf_file_url"]
                    + " download failed on "
                    + str(datetime.today().date())
                )
                logText += "\n " + traceback.format_exc()
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                return False

            # scrapeScript: ScrapScript = get_scrap_script_by_file_name(mysql_driver, os.path.basename(__file__))
            scrapeScript: ScrapScript = get_scrape_script_by_scraperUrlId(
                mysql_driver, scrapeURLId
            )
            document = StaticPublicDocument(
                number=df["Number"],
                title=df["title"],
                description=df["description"],
                url=df["pdf_file_url"],
                # documentType=df['filetype'],
                documentType=scrapeScript.documentTypeID,
                documentStatus=1,  # Active
                activeDate=date(df["curr_year"], 1, 1),
                inactiveDate=None,
                sourceFileName=os.path.relpath(path, config.downloadDir),
                pdfFileName=os.path.relpath(pdf_path, config.downloadDir),
                parsed=False,
                embedded=False,
                parsingScriptID=scrapeScript.defaultParsingScriptID,
                createdDate=datetime.today().date(),
                modifiedDate=datetime.today().date(),
                # additionalInfo=df['pdf_filename'],
                fileSize=os.path.getsize(pdf_path),
                parsingLog="notParsed",
                embeddingLog="notEmbedded",
                noOfParagraphs=0,
                lastScrapeDate=datetime.today().date(),
                # scrapingLog = 'scraped successfully on ' + str(datetime.today().date()),
                sourceProject = 0,
            )

            # insert document in DB
            insert_document(mysql_driver, document, doc_class=StaticPublicDocument)
            logText = (
                df["pdf_filename"]
                + " with filetype = "
                + str(df["filetype"])
                + " at "
                + df["pdf_file_url"]
                + " downloaded on "
                + str(datetime.today().date())
            )
            logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)

        else:
            docInDB.lastScrapeDate = datetime.today().date()
            # docInDB.scrapingLog = 'File ' + df['title']  + ' with CFR Doc at ' + df['pdf_file_url'] + ' scraped and not changed on ' + str(datetime.today().date())
            logText = (
                "File "
                + df["title"]
                + " with CFR Doc at "
                + df["pdf_file_url"]
                + " scraped and not changed on "
                + str(datetime.today().date())
            )
            logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)

            # update document in DB
            update_documents(mysql_driver, [docInDB])
    except Exception:
        logText = (
            "ERROR in Scraping File "
            + df["title"]
            + " with CFR Doc at "
            + df["pdf_file_url"]
            + " on "
            + str(datetime.today().date())
        )
        logText += "\n " + traceback.format_exc()
        logList.append(logText)
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)
        print(logText)
        return False

    return True


def cancel_previous_editions(
    config: ScriptsConfig, mysql_driver: MySQLDriver, df, scrapeURLId
):
    global logList
    try:
        df["xml_file_url"]
        PDF_URL = df["pdf_file_url"]

        curr_year = df["curr_year"]

        years_to_check = curr_year - 2020

        if years_to_check <= 0:
            return

        for y in range(1, years_to_check):
            pdf_url_to_check = PDF_URL.replace(str(curr_year), str(curr_year - y))

            # check if document exists in documents table
            docInDB = find_document_by_url(mysql_driver, pdf_url_to_check, doc_class=StaticPublicDocument)

            if docInDB:
                # Mark as Cancelled/Archived
                docInDB.documentStatus = 3  # 3 Cancelled
                docInDB.inactiveDate = datetime.today().date()
                docInDB.modifiedDate = datetime.today().date()

                # update in DB
                cancel_documents(mysql_driver, [docInDB])

                # write to log
                logText = (
                    "File "
                    + docInDB.pdfFileName
                    + " with "
                    + str(docInDB.documentType)
                    + " changed to cancelled on "
                    + str(datetime.today().date())
                )
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                return True

    except Exception:
        logText = (
            df["pdf_filename"]
            + " with filetype = "
            + str(df["filetype"])
            + " at "
            + df["pdf_file_url"]
            + " download failed on "
            + str(datetime.today().date())
        )
        logText += "\n " + traceback.format_exc()
        logList.append(logText)
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)
        print(logText)
        return False


def save_log_data_to_db(log_list, mysql_driver, scrapeURLId):
    # Passing ScrapeURL ID hard-coded as it is already declared and won't be changed
    update_scrape_url_set_log_value(mysql_driver, scrapeURLId, json.dumps(log_list))


def run(config: ScriptsConfig, scrapeURLId: int):
    global logList
    scraped_df = get_fda_cfrs()

    # check for new documents
    mysql_driver = MySQLDriver(cred=config.databaseConfig.__dict__)

    for idx, row in scraped_df.iterrows():
        status = download_cfr_document(config, mysql_driver, row, scrapeURLId)

        if status is True:
            cancel_previous_editions(config, mysql_driver, row, scrapeURLId)
        else:
            continue

    # create a function and save he logs
    #    save_log_data_to_db(logList, mysql_driver, scrapeURLId)
    print("Scraped CFR FDA documents")

def parse_remaining_args(cleaned_args):
    repo_id = None
    values = []

    i = 0
    while i < len(cleaned_args):
        if cleaned_args[i] == '--repo_id':
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
        #configure the logging level
        remaining_args = configure_logging_from_argv(default_level='INFO')
        repo_id, docIdsList = parse_remaining_args(remaining_args)

        if len(docIdsList) > 0:
            scrapeURLId = docIdsList[0]
        else:
            scrapeURLId = 1

        configs = parseCredentialFile("/app/tlp_config.json")

        if configs:
            run(configs, scrapeURLId, repo_id)
    except Exception as e:
        print("The EXCEPTION >>>>>>>>>>>>>> ")
        print(traceback.format_exc())
        traceback.print_exc()
        print(e)
