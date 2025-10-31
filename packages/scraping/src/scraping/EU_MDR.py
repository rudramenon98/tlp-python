## Library import
import io
import json
import logging
import os
import sys
import time
import traceback
import urllib.request
from datetime import date, datetime
from multiprocessing import Pool, cpu_count, current_process

import pandas as pd
import requests
from common_tools.log_config import configure_logging_from_argv
from database.document_service import (
    cancel_documents,
    find_document_by_url,
    find_documents_not_scraped_on_date,
    get_scrape_script_by_scraperUrlId,
    insert_documents_bulk2,
    update_documents,
)
from database.entity.Document import Document
from database.entity.ScrapScript import ScrapScript
from database.entity.ScriptsProperty import ScriptsConfig, parseCredentialFile
from database.scrape_url_service import (
    scrape_url_append_log,
    update_scrape_url_set_log_value,
)
from database.utils.MySQLFactory import MySQLDriver
from database.utils.util import get_dir_safe
from database.utils.WebDriverFactory import WebDriverFactory
from PyPDF2 import PdfReader
from selenium.webdriver.support.ui import WebDriverWait

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
'''
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Console (stdout) handler
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
'''

logList = []
DateToday = datetime.today()
DefaultActiveDate = date(DateToday.year, 1, 1)


def scrap_table(driver, url):
    
    i = 2
    cn = 0
    dataset = {'Number':[],'title':[],'description':[],'pdf_file_name':[],'pdf_file_url':[],'filetype':[],'status':[],'active_date':[],'Download':[]}
    while True:
        try:
            
            tr = url+f'[{i}]'
            #tr = '//*[@id="content"]/section[1]/div/div/div[1]/div/figure[1]/table/tbody/tr'+f'[{i}]'
            #print('//*[@id="content"]/section[1]/div/div/div[1]/div/figure[1]/table/tbody/tr'+f'[{i}]')
            table_trs = driver.find_element_by_xpath(tr)
            td = [driver.find_element_by_xpath(tr+'/'+'td'+f'[{i}]').text for i in range(1,4) ]
            dataset['title'].append(td[1])
            
            link = driver.find_element_by_xpath(tr+'/'+'td[2]/a').get_attribute('href')
            dataset['pdf_file_url'].append(link)
            
            file_name = link.split('/')[-1]
            log.debug("Processing file: %s", file_name)
            dataset['pdf_file_name'].append(file_name)
            
            
            dataset['description'].append(td[0])
            
            dataset['filetype'].append(5)
            
            dataset['Download'].append(True)
            
            dataset['status'].append('Active') 
            
            dataset['active_date'].append(DefaultActiveDate)
            
            dataset['Number'].append(cn)
            
            i+=1
            cn+=1
        except:
            break
    df  = pd.DataFrame(dataset)
    return df


def download_file_low(URL, path):
    hdr = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9,gu;q=0.8,hi;q=0.7',
        'Connection': 'keep-alive'
    }
    chunk_size = (1024 * 1024) * 1  # 1MB

    response = requests.get(URL, headers=hdr, stream=True)
    total = int(response.headers.get('content-length', 0))

    with open(path, 'wb') as file:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)


def download_file(config: ScriptsConfig, document: Document, sequence):
    try:
        #mysql_driver = MySQLDriver(cred=config.main_databaseConfig.__dict__)
        #file_name = document.title[:50].replace('/',' ') + ".pdf"
        
        #Try to preserve the filename from the website
        file_name = document.url.split('/')[-1]

        if file_name.lower() == 'native':  
            #Construct a filename as sometimes the Website URL has only 'native' at the end of the URL
            file_name = document.title[:50].replace('/',' ') + ".pdf"

        #dir_name = config.rootDataDir + config.downloadDirInRootDir
        dir_name = config.downloadDir
        path = get_dir_safe(dir_name) + "/" + file_name
        log.debug("%s -> Downloading file %s url:%s on location:%s", current_process(), sequence, document.url, path)
        #response = urllib.request.urlopen(document.url)
        #file = open(path, 'wb')
        #file.write(response.read())
        #file.close()
        download_file_low(document.url, path)
        document.pdfFileName = os.path.relpath(path, config.downloadDir)
        document.fileSize = os.path.getsize(path)
        document.parsed = False
        #insert_document(mysql_driver, document)
        #mysql_driver.close()
        return document, 'Success'
    except Exception as ex:
        log.debug('File download failed: %s', document.title + ".pdf")
        log.debug("%s", ex)
        traceback.print_exc()
        return None, str(traceback.format_exc())        


def scrape_main_page(main_URL, docker_url):

    #chrome_options = Options()
    #chrome_options.add_argument("user-agent=whatever you want")
    #chrome_options.add_argument('--no-sandbox')
    #chrome_options.add_argument('--disable-dev-shm-usage')
    #chrome_options.add_argument("--headless")

    #driver = WebDriverFactory.getWebDriverInstance(browser='chrome')
    driver = WebDriverFactory.getWebDriverInstance(browser='docker', docker_url=docker_url)

    # wait for the page to fully load
    wait = WebDriverWait(driver, 2)

    # navigate to the main_URL
    driver.get(main_URL)
    wait = WebDriverWait(driver, 20)
    time.sleep(20)

    #'british'
    url2 = '//*[@id="content"]/section[1]/div/div/div[1]/div/div[1]/div/div/div/div/div/figure/table/tbody/tr'
    df_british = scrap_table(driver, url2)

    df_british = df_british.assign(filetype = 6)

    #'main'
    url ='//*[@id="content"]/section[1]/div/div/div[1]/div/figure/table/tbody/tr'
    df_main = scrap_table(driver, url)

    #'tuv'
    url3 = '//*[@id="content"]/section[1]/div/div/div[1]/div/div[2]/div/figure/table/tbody/tr'
    df_tuv = scrap_table(driver, url3)
    df_tuv = df_tuv.assign(filetype = 7)

    #'other'
    url4 = '//*[@id="content"]/section[1]/div/div/div[1]/div/figure[2]/table/tbody/tr'
    df_other= scrap_table(driver, url4)
    df_other = df_other.assign(filetype = 8)

    final_df = pd.concat([df_main,df_british,df_tuv,df_other],ignore_index=True)

    driver.close()

    return final_df


def check_for_new_documents(mysql_driver, scrapeDF, scrapeURLId, scrapeScript: ScrapScript):
    #global logList
    download_list = []
    update_list = []
    skip_list = []

    for idx, row in scrapeDF.iterrows():

        file_url = row['pdf_file_url']

        try:
#            if row['Number'] == 0:
#                continue

            if isinstance(file_url, list):
                if len(file_url) > 0:
                    file_url = str(file_url[0])
                else:
                    skip_list.append(row['Number'] + ' ' + row['title'])
                    logText = 'Skipped File ' + row['title'] + ' with AC at ' + row[
                        'Number'] + ' URL = <null>  on ' + str(datetime.today())
                    #logList.append(logText)
                    scrape_url_append_log(mysql_driver, scrapeURLId, logText)                    
                    log.debug("%s", logText)
                    continue
            
            docInDB = find_document_by_url(mysql_driver, file_url)
        except Exception as excep:
            logText = f'Failed for : {file_url} \n'
            logText += traceback.format_exc()
            #logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            log.debug("%s", logText)                
            continue

        #scrapeScript: ScrapScript = get_scrap_script_by_file_name(mysql_driver, os.path.basename(__file__))

        if not docInDB:
            log.debug('Document not found in DB: %s', file_url)
            try:
                document = Document(
                    number=row['Number'],
                    title=row['title'],
                    description=row['description'][:1024],
                    url=row['pdf_file_url'],
                    documentType=row['filetype'],
                    #documentType=scrapeScript.documentTypeID,
                    documentStatus=1,  # Active
                    activeDate=row['active_date'],
                    inactiveDate=None,
                    sourceFileName=None,
                    pdfFileName=None,
                    parsed=False,
                    embedded=False,
                    #parsingScriptID=scrapeScript.defaultParsingScriptID,
                    parsingScriptID=scrapeScript.defaultParsingScriptID,
                    createdDate=datetime.today().date(),
                    modifiedDate=datetime.today().date(),
                    # additionalInfo=row['pdf_file_name'],
                    fileSize=0,
                    parsingLog='notParsed',
                    embeddingLog='notEmbedded',
                    noOfParagraphs=0,
                    lastScrapeDate=datetime.today().date(),
                    # scrapingLog = 'scraped successfully',
                    sourceProject=0,
                )
                download_list.append(document)
            except Exception as exc:
                logText = f'New Document row creation failer for : {file_url} \n'
                logText += traceback.format_exc()
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                log.debug("%s", logText)                

        else:
            log.debug('Document found in DB: %s', file_url)
            docInDB.lastScrapeDate = datetime.today().date()
            logText = 'File ' + row['title'] + ' with Guidance Doc at ' + row[
                'pdf_file_url'] + ' scraped and not changed on ' + str(datetime.today())
            logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            log.debug("%s", logText)                
            update_list.append(docInDB)

    return update_list, download_list


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


def check_for_cancelled_documents(mysql_driver, current_date, scrapeURLId):
    global logList

    cancel_list = []

    # get the documents in documents table which were not scraped today
    old_documents_list = find_documents_not_scraped_on_date(mysql_driver, current_date)

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

    main_URL = 'https://www.medical-device-regulation.eu/mdr-guidance-documents/'

    logText = f'ScrapeID: {scrapeURLId}: Scrape for MDR Guidance Documents started at {datetime.today()} URL: {main_URL}'

    #logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                log.debug("%s", logText)

    # get all the documents and their details
    scrapeDF = scrape_main_page(main_URL, config.SeleniumDocker)

    logText = f'Website data captured at {datetime.today()}' 
    #logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText) 
                log.debug("%s", logText)

    update_list, download_list = check_for_new_documents(mysql_driver, scrapeDF, scrapeURLId, scrapeScript)

    #update the unchanged documents in DB (just set their lastScrapeDate = DateToday)
    if update_list and len(update_list) > 0:
        update_documents(mysql_driver, update_list)
    
    print("final download  list size:")
    print(download_list)
    print(len(download_list))
    
    bulk_document = []
    logText = f'File downloading started at {datetime.today()}'
    #logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)

    t1 = time.time()
    print("starting download files")
    cnt = 0
    for doc in download_list:
        print("DOC:", str(doc.number))
        cnt += 1
        docs, msgText = download_file(config,doc,cnt)
        if docs is not None:
            bulk_document.append(docs)
        else:
            logList.append(msgText)
            scrape_url_append_log(mysql_driver, scrapeURLId, msgText)    


    logText = f'File downloading completed at {datetime.today()} time taken: {str((time.time() - t1))}' 
    #logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                log.debug("%s", logText)

    docIDsList = None
    try:
        if bulk_document is not None and len(bulk_document) > 0:
            docIDsList = insert_documents_bulk2(mysql_driver, bulk_document)

        if docIDsList is not None and len(docIDsList) > 0:
            logText = f'Newly added docIDs: {docIDsList}'
            #logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)    
            log.debug("%s", logText)
    except:
        logText = traceback.format_exc()
        #logList.append(logText)
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                log.debug("%s", logText)

    # handle cancelled documents
    cancel_list = check_for_cancelled_documents(mysql_driver, DateToday, scrapeURLId)

    logText = f'Number of cancelled documents since last scrape: {len(cancel_list)}'
    logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                log.debug("%s", logText)

    # Mark the newly cancelled documents in the documents table
    if cancel_list and len(cancel_list) > 0:
        cancel_documents(mysql_driver, cancel_list)


    logText = f'Scraping URL: {main_URL} DONE'
    #logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)    
                log.debug("%s", logText)

    #save_log_data_to_db(logList, mysql_driver)
    print('Scaped URL:' + main_URL)




if __name__ == '__main__':
    try:
        props = None
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
            scrapeURLId = 2

        configs = parseCredentialFile('/app/tlp_config.json')
        #configs = parseCredentialFile('//dockers/Enginius/test/scripts/testmed-tlp_config.json')
        if configs:
            run(configs, scrapeURLId)
    except Exception as e:
        print('The EXCEPTION >>>>>>>>>>>>>> ')
        print(traceback.format_exc())
        traceback.print_exc()
        print(e)
        
