import io
import json
import logging
import os
import sys
import time
import traceback
import urllib.request
from datetime import datetime
from multiprocessing import Pool, cpu_count, current_process

import pandas as pd
import requests
from app.document_service import (
    cancel_documents,
    find_document_by_url,
    find_documents_not_scraped_on_date,
    find_documents_not_scraped_on_date_by_type,
    get_scrap_script_by_file_name,
    get_scrape_script_by_scraperUrlId,
    insert_document,
    insert_documents_bulk2,
    update_documents,
)
from app.entity.Document import Document
from app.entity.ScrapScript import ScrapScript
from app.entity.ScriptsProperty import ScriptsConfig, parseCredentialFile
from app.scrape_url_service import (
    scrape_url_append_log,
    update_scrape_url_set_log_value,
)
from app.utils.MySQLFactory import MySQLDriver
from app.utils.util import get_dir_safe
from app.utils.WebDriverFactory import WebDriverFactory
from dateutil.parser import parse
from logconfig import configure_logging_from_argv
from PyPDF2 import PdfReader
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
'''
# Console (stdout) handler
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
'''

logList = [] 
DateToday = datetime.today()
from urllib import request


def download_file(URL, path):
    hdr = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9,gu;q=0.8,hi;q=0.7',
        'Connection': 'keep-alive'
    }
    chunk_size = (1024 * 1024) * 1  # 1MB

    response = requests.get(URL, headers=hdr,verify = '/app/scripts/venv/lib/python3.9/site-packages/certifi/cacert.pem')
    total = int(response.headers.get('content-length', 0))
    #filename = request.urlopen(request.Request(URL)).info().get_filename()
    filename = URL.split('/')[-1]
    #path = path+'/'+filename
    with open(path, 'wb') as file, tqdm(
            desc=path,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
            
   
def Canada_Guidance_Scraping(url, docker_url):
    driver = WebDriverFactory.getWebDriverInstance(browser='docker', docker_url=docker_url)
    #driver.get('https://laws-lois.justice.gc.ca/eng/regulations/SOR-98-282/')
    driver.get(url)
    all_list_elements = driver.find_elements(By.XPATH,'/html/body/div/div/main/div[1]/header')
    dataset = {'Number':[],'title':[],'discription':[],'pdf_file_name':[],'pdf_file_url':[],'filetype':[],'status':[],'active_date':[],'Download':[]}
    cn = 0
    for i in all_list_elements:
        texts = i.text.split('\n')
        log.debug("%s", texts)
        links =[k.get_attribute('href') for k in  i.find_elements_by_tag_name('a')][:3]
        dataset['title'].append(texts[0])
        dataset['pdf_file_url'].append(links)
        dataset['discription'].append(texts[8])
        dt, tokens = parse(texts[8], fuzzy_with_tokens=True)
        #date = dt.strftime("%m/%d/%Y")
        date = dt.date()
        file_name = links[-1].split('/')[-1]
        log.debug("Processing file: %s", file_name)
        dataset['pdf_file_name'].append(file_name)
        dataset['filetype'].append(9)
        dataset['Download'].append(True)
        dataset['status'].append('Active') 
        dataset['active_date'].append(date)
        dataset['Number'].append(cn)
        cn+=1
    df = pd.DataFrame.from_dict(dataset, orient='index')
    df = df.transpose()
    return df 


def check_for_new_documents(config: ScriptsConfig, mysql_driver, scrapeDF, scrapeURLId, scrapeScript: ScrapScript):
    #global logList
    skip_list = []

    for idx, row in scrapeDF.iterrows():

        file_url = row['pdf_file_url']

        try:
#            if row['Number'] == 0:
#                continue

            if isinstance(file_url, list):
                if len(file_url) > 0:
                    for u in file_url:
                        if str(u).lower().endswith('.pdf'):
                            PDF_URL = str(u)
                        elif str(u).lower().endswith('.html'):
                            HTML_URL = str(u)
                else:
                    skip_list.append(row['Number'] + ' ' + row['title'])
                    logText = 'Skipped File ' + row['title'] + ' with AC at ' + row[
                        'Number'] + ' URL = <null>  on ' + str(datetime.today())
                    #logList.append(logText)
                    scrape_url_append_log(mysql_driver, scrapeURLId, logText)                    
                    log.debug("%s", logText)
                    continue
            
            docInDB = find_document_by_url(mysql_driver, PDF_URL)
        except Exception as excep:
            logText = f'Failed for : {file_url} \n'
            logText += traceback.format_exc()
            #logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            log.debug("%s", logText)                
            continue

        #scrapeScript: ScrapScript = get_scrap_script_by_file_name(mysql_driver, os.path.basename(__file__))

        if not docInDB:
            log.debug('Document not found in DB: %s', PDF_URL)
            src_file_name = HTML_URL.split("/")[-1]
            pdf_file_name = PDF_URL.split("/")[-1]

            #folder = config.rootDataDir + config.downloadDirInRootDir
            folder = config.downloadDir
            path = get_dir_safe(folder) + "/" + src_file_name
            pdf_path = get_dir_safe(folder) + "/" + pdf_file_name

            try:
                download_file(HTML_URL, path)
            except Exception as ex1:
                logText = src_file_name + ' with filetype = ' + str(scrapeDF['filetype']) + ' at ' + HTML_URL + ' download failed on ' + str(datetime.today().date())
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)                
                log.debug("%s", traceback.format_exc())
                return False

            try:
                download_file(PDF_URL, pdf_path)
            except Exception as ex2:
                logText = pdf_file_name + ' with filetype = ' + str(scrapeDF['filetype']) + ' at ' + PDF_URL  + ' download failed on ' + str(datetime.today().date())
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)                
                log.debug("%s", traceback.format_exc())
                return False

            try:
                document = Document(
                    number=row['Number'],
                    title=row['title'],
                    description=row['discription'][:1024],
                    #url=row['pdf_file_url'],
                    url=PDF_URL,
                    #documentType=row['filetype'],
                    documentType=scrapeScript.documentTypeID,
                    documentStatus=1,  # Active
                    activeDate=row['active_date'],
                    inactiveDate=None,
                    sourceFileName=src_file_name,
                    pdfFileName=pdf_file_name,
                    parsed=False,
                    embedded=False,
                    #parsingScriptID=scrapeScript.defaultParsingScriptID,
                    parsingScriptID=scrapeScript.defaultParsingScriptID,
                    createdDate=datetime.today().date(),
                    modifiedDate=datetime.today().date(),

                    fileSize=os.path.getsize(pdf_path),
                    parsingLog='notParsed',
                    embeddingLog='notEmbedded',
                    noOfParagraphs=0,
                    lastScrapeDate=datetime.today().date(),
                    # scrapingLog = 'scraped successfully',
                    sourceProject = 0,
                )
#                download_list.append(document)
                insert_document(mysql_driver, document)
            except Exception as exc:
                logText = f'New Document row creation failer for : {file_url} \n'
                logText += traceback.format_exc()
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                log.debug("%s", logText)
                return False

        else:
            log.debug('Document found in DB: %s', file_url)
            docInDB.lastScrapeDate = datetime.today().date()
            logText = 'File ' + row['title'] + ' with Guidance Doc at ' + str(PDF_URL) + ' scraped and not changed on ' + str(datetime.today())
            logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            log.debug("%s", logText)
            #update_list.append(docInDB)
            update_documents(mysql_driver, [docInDB])

    return True


def check_if_file_exists3(link, mysql_driver, scrapeURLId):
    # link = 'https://www.govinfo.gov/content/pkg/CFR-'+str(year)+'-title'+str(title)+'-vol'+str(vol)+'/pdf/CFR-'+str(year)+'-title'+str(title)+'-vol'+str(vol)+'.pdf'
    # print(link)
    hdr = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9,gu;q=0.8,hi;q=0.7',
        'Connection': 'keep-alive'
    }

    # page = requests.get(str(link), headers=hdr)

    chunk_size = (1024 * 1024) * 1  # 1MB

    response = requests.get(str(link), headers=hdr, stream=True)
    total = int(response.headers.get('content-length', 0))

    with io.BytesIO() as stream:
        size = 0
        for data in response.iter_content(chunk_size=chunk_size):
            size += stream.write(data)

        if total != size:
            print('Warning ' + str(link) + ' Downladed ' + str(size) + 'out of ' + str(total))
            logText = 'Warning ' + str(link) + ' Downladed ' + str(size) + 'out of ' + str(total)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
        try:
            pdf = PdfReader(stream)
            # print(pdf)
            return True
        except:
            # print('Year',year,'Title',title,'Vol',vol,'is not found')
            return False

    return False


def check_for_cancelled_documents(mysql_driver, current_date, scrapeURLId, scrapeScript: ScrapScript):
    global logList

    cancel_list = []

    # get the documents in documents table which were not scraped today
    #old_documents_list = find_documents_not_scraped_on_date(mysql_driver, current_date)
    old_documents_list = find_documents_not_scraped_on_date_by_type(mysql_driver, current_date, document_type = scrapeScript.documentTypeID)
    if not old_documents_list or len(old_documents_list) == 0:
        return None

    for old_doc in old_documents_list:
        old_number = old_doc.number
        old_url = old_doc.url

        # check if url exists, i.e. guidance document is downloadable from FDA website
        doc_exists = check_if_file_exists3(old_url, mysql_driver, scrapeURLId)

        if doc_exists == False:
            # Mark as Inactive
            old_doc.documentStatus = 2
            old_doc.inactiveDate = datetime.today().date()
            old_doc.modifiedDate = datetime.today().date()

            # write to log
            logText = 'File ' + str(old_doc.pdfFileName )+ ' with ' + str(old_doc.documentType )+ ' changed to withdrawn/cancelled on ' + str(
                datetime.today())
            logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)                
            log.debug("%s", logText)

            # append to cancel list
            cancel_list.append(old_doc)

    return cancel_list

def run(config: ScriptsConfig, scrapeURLId):
    #global logList
    
    DateToday = datetime.today().date()
    mysql_driver = MySQLDriver(cred=config.databaseConfig.__dict__)
    scrapeScript: ScrapScript = get_scrape_script_by_scraperUrlId(mysql_driver, scrapeURLId)

    main_URL = 'https://laws-lois.justice.gc.ca/eng/regulations/SOR-98-282/'

    logText = f'ScrapeID: {scrapeURLId}: Scrape for Canada Guidance Documents started at {datetime.today()} URL: {main_URL}'

    #logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                log.debug("%s", logText)

    # get all the documents and their details
    scrapeDF = Canada_Guidance_Scraping(main_URL, config.SeleniumDocker)

    logText = f'Website data captured at {datetime.today()}' 
    #logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText) 
                log.debug("%s", logText)

    retval = check_for_new_documents(config, mysql_driver, scrapeDF, scrapeURLId, scrapeScript=scrapeScript)

    # handle cancelled documents
    cancel_list = check_for_cancelled_documents(mysql_driver, DateToday, scrapeURLId, scrapeScript=scrapeScript)


    # Mark the newly cancelled documents in the documents table
    if cancel_list and len(cancel_list) > 0:
        logText = f'Number of cancelled documents since last scrape: {len(cancel_list)}'
        logList.append(logText)
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                log.debug("%s", logText)

        cancel_documents(mysql_driver, cancel_list)

    logText = f'Scraping URL: {main_URL} DONE'
    #logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)    
                log.debug("%s", logText)

    #save_log_data_to_db(logList, mysql_driver)
    print('Scaped URL:' + main_URL)




if __name__ == '__main__':
    try:
        #configure the logging level
        remaining_args = configure_logging_from_argv(default_level='INFO')

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

        configs = parseCredentialFile('/app/tlp_config.json')

        if configs:
            run(configs, scrapeURLId)
    except Exception as e:
        print('The EXCEPTION >>>>>>>>>>>>>> ')
        print(traceback.format_exc())
        traceback.print_exc()
        print(e)

