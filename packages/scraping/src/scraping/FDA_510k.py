## Library import
import io
import json
import os
import shutil
import random
import csv
import sys
import time
import requests, zipfile, io
import pandas as pd
import lxml.html as lh
import urllib
import requests
from bs4 import BeautifulSoup

import traceback
import urllib.request

from datetime import datetime
from multiprocessing import current_process, Pool, cpu_count

import pandas as pd
import requests
from PyPDF2 import PdfReader

from database.document_service import insert_document, insert_documents_bulk2, find_document_by_number, update_documents, \
    find_documents_not_scraped_on_date, get_scrapeurl_by_scrapeUrlId, get_scrap_script_by_file_name, cancel_documents, \
    get_scrape_script_by_scraperUrlId, update_documents2
from database.entity.K510Records import K510Record
from database.entity.ScrapScript import ScrapScript
from database.entity.ScriptsProperty import ScriptsConfig, parseCredentialFile
from database.scrape_url_service import update_scrape_url_set_log_value, scrape_url_append_log
from database.utils.MySQLFactory import MySQLDriver
from database.utils.WebDriverFactory import WebDriverFactory
from database.utils.util import get_dir_safe

import logging
from common_tools.log_config import configure_logging_from_argv

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def Error_Handler(fn):
    def Inner_Function(*args, **kwargs):
        try:
            t1 = time.time()
            ret = fn(*args, **kwargs)
            t2 = time.time()
            log.debug(f'Function {fn.__name__} executed in {(t2-t1):.4f}s')
            return ret
        except Exception as exc:
            log.error(f"{fn.__name__}")
            traceback.print_exc()
    return Inner_Function

# for logging
logList = []
DateToday = datetime.today()

@Error_Handler
def get_main_page(main_URL, downloadDir):
    
    #Run this for first time

    #main_URL = 'https://www.accessdata.fda.gov/premarket/ftparea/pmn96cur.zip'
    try:
        hdr = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9,gu;q=0.8,hi;q=0.7',
            'Connection': 'keep-alive'
        }

        fda_file = requests.get(main_URL, headers=hdr)
        zip_file_download = zipfile.ZipFile(io.BytesIO(fda_file.content))

        zip_file_path  = get_dir_safe(downloadDir + "/510k/")

        #clean folder before extracting the zip file
        dir_path = zip_file_path
        dir_list = os.listdir(dir_path)
        for item in dir_list:
            if item.endswith(".txt"):
                print(f"Removing file: {os.path.join(dir_path,item)}")
                #os.remove(os.path.join(dir_path, item))

        # Extract the zip file
        zip_file_download.extractall(zip_file_path)

        pm_filename = ''
        if os.path.isfile(zip_file_path+'/pmn96cur.txt'):
            pm_filename = zip_file_path+'/pmn96cur.txt'
        elif os.path.isfile(zip_file_path+'pmnistmn.txt'):
            pm_filename = zip_file_path+'pmnistmn.txt'     

        #with open(zip_file_path+'/pmn96cur.txt', encoding='ISO-8859-1') as fda_510_text_file: 
        with open(pm_filename, encoding='ISO-8859-1') as fda_510_text_file: 
            fda_510_text_file_lines = fda_510_text_file.readlines()
            fda_510_text_file_lines = [i.strip('\n') for i in fda_510_text_file_lines]

            return fda_510_text_file_lines, ''
    except Exception as ex:
        err_msg = traceback.format_exc()
        return None, err_msg

Document_ID = []
Number = []
Title = []
Active_Date = []
Document_Type = []
Document_Status = []
#LastScrapeDate=[]
SourceFileName =[]
Description  = []
Summary_link = []
pdf_url = []

@Error_Handler   
def scrape_510K_records(config: ScriptsConfig, fda_510k_text_files, mysql_driver, scrapeURLId):
    #dfs = []
    
    dataset = {

            'Number':[],
            'title':[],
            'description':[],
            'pdf_file_url':[],
            'pdf_file_name':[],
            'filetype':[],
            'status':[],
            'active_date':[],
            'Number':[],
            'applicant':[],
            'contact':[],
            'receivedDate':[],
            'decisionDate':[],
            'decisionCode':[],
            'reviewAdvisoryCommittee':[],
            'productCode':[],
            "stmtOrsummary":[],
            'classificationAdvisoryCommittee':[],
            'type':[],
            'thirdPartyFlag':[],
            'expeditedReviewFlag':[],
            'DeviceName':[],
            'filetype':[],
            'status':[],
            'url':[]
    }

    url_prefix = 'https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID='

    try:

        count = 0
        for row in fda_510k_text_files:

            if row['Number'].startswith("D"):
                continue
            
            count += 1
        #    if count > 10:
        #        break

            dataset['Number'].append(row['Number'])
            dataset['title'].append(row['title'])
            dataset['description'].append(row['description'])
            dataset['pdf_file_url'].append(row['pdf_file_url'])
            dataset['pdf_file_name'].append(row['pdf_file_name'])
            dataset['filetype'].append(10) #scrapeScript.documentTypeID,
            dataset['status'].append(2) #Inactive till PDF is downloaded
            dataset['active_date'].append(datetime.strptime(row['active_date'], "%m/%d/%Y").date())
            dataset['Number'].append(row['Number'])
            dataset['applicant'].append(row['applicant'])
            dataset['contact'].append(row['contact'])
            dataset['receivedDate'].append(datetime.strptime(row['receivedDate'], "%m/%d/%Y").date())
            dataset['decisionDate'].append(datetime.strptime(row['decisionDate'], "%m/%d/%Y").date())
            dataset['decisionCode'].append(row['decisionCode'])
            dataset['reviewAdvisoryCommittee'].append(row['reviewAdvisoryCommittee'])
            dataset['productCode'].append(row['productCode'])
            dataset["stmtOrsummary"].append(row["stmtOrsummary"])
            dataset['classificationAdvisoryCommittee'].append(row['classificationAdvisoryCommittee'])
            dataset['type'].append(row['type'])
            dataset['thirdPartyFlag'].append(row['thirdPartyFlag'])
            dataset['expeditedReviewFlag'].append(row['expeditedReviewFlag'])
            dataset['DeviceName'].append(row['DeviceName'])
            dataset['filetype'].append(10) #scrapeScript.documentTypeID,
            dataset['status'].append(2) #Inactive till PDF is downloaded
            dataset['url'].append(f'{url_prefix}{row['Number']}')

    except:
            logText = traceback.format_exc()
            logList.append(logText)
            print(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)

    df = pd.DataFrame.from_dict(dataset, orient='index')
    df = df.transpose()
    return df


@Error_Handler
def find_510K_record_by_k_number(mysql_driver, number):
    session = mysql_driver.get_session()
    result = session.query(K510Record) \
        .filter(K510Record.k_number == number) \
        .first()
    session.close()
    return result

def insert_510K_records(config, mysql_driver, scrapeDF, scrapeURLId, scrapeScript: ScrapScript):
    global logList
    download_list = []
    update_list = []
    skip_list = []

    for idx, row in scrapeDF.iterrows():

        file_url = row['pdf_file_url']

        k_number = row['Number'].strip()
        try:
            if len(k_number) == 0:
                skip_list.append(row['Number'] + ' ' + row['title'])
                logText = 'Skipped File ' + row['title'] + ' with Premarket Notificaiton at ' + row[
                    'Number'] + ' URL = <null>  on ' + str(datetime.today())
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)                    
                print(logText)
                continue

            docInDB = find_510K_record_by_k_number(mysql_driver, k_number)

        except Exception as excep:
            logText = f'Failed for : {file_url} \n'
            logText += traceback.format_exc()
            logList.append(logText)
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            print(logText)                
            continue

        #scrapeScript: ScrapScript = get_scrap_script_by_file_name(mysql_driver, os.path.basename(__file__))

        if not docInDB:
            try:
                record = K510Record(
                    k_number=row['Number'],
                    applicant=row['applicant'],
                    contact=row['contact'],
                    receivedDate=row['receivedDate'],
                    decisionDate=row['decisionDate'],
                    decisionCode=row['decisionCode'],
                    reviewAdvisoryCommittee=row['reviewAdvisoryCommittee'],
                    productCode=row['productCode'],
                    statementOrSummary=row["stmtOrsummary"],
                    classificationAdvisoryCommittee=row['classificationAdvisoryCommittee'],
                    Type=row['type'],
                    thirdPartyFlag=row['thirdPartyFlag'],
                    expeditedReviewFlag="N",
                    DeviceName="Super Heart Monitor",
                    lastScrapeDate=datetime.today().date()
                )   
                            
                download_list.append(record)
                logText = '510(k) Record ' + row['title'] + ' with FDA 510(k) Premarket Notifications scraped on ' + str(datetime.today())
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                print(logText)

                
            except Exception as exc:
                logText = f'New Document row creation failed for : {file_url} \n'
                logText += traceback.format_exc()
                logList.append(logText)
                scrape_url_append_log(mysql_driver, scrapeURLId, logText)
                print(logText)                

        else:
            record = K510Record(
                k_number=row['Number'],
                applicant="Acme Medical Devices",
                contact="John Doe",
                receivedDate="2023-01-15",
                decisionDate="2023-03-01",
                decisionCode="A510",
                reviewAdvisoryCommittee="OR",
                productCode="XYZ",
                statementOrSummary="Summary",
                classificationAdvisoryCommittee="NE",
                Type="Traditional",
                thirdPartyFlag=True,
                expeditedReviewFlag="N",
                DeviceName="Super Heart Monitor",
                lastScrapeDate=datetime.today().date()
            )
            docInDB.lastScrapeDate = datetime.today().date()
            logText = '510(K) Record ' + row['title'] + ' with FDA 510(k) Premarket Notifications scraped and not changed on ' + str(datetime.today())
            logList.append(logText)            
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            print(logText)                

            update_list.append(docInDB)

    return update_list, download_list

def run(config: ScriptsConfig, scrapeURLId):
    global logList
    DateToday = datetime.today().date()

    main_URL = 'https://www.accessdata.fda.gov/premarket/ftparea/pmn96cur.zip'

    downloadDir = config.downloadDir

    mysql_driver = MySQLDriver(cred=config.main_databaseConfig.__dict__)

    logText = f'Scrape for FDA 510(k) Premarket Notifications started at {datetime.today()} URL: {main_URL}'
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)
    print(logText)
    
    scrapeURL = get_scrapeurl_by_scrapeUrlId(mysql_driver, scrapeURLId)
    scrapeScript: ScrapScript = get_scrape_script_by_scraperUrlId(mysql_driver, scrapeURLId)
#                url = f'https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID={k_value}'

    if scrapeURL is None:
        # should never reach here!
        logText = f'ERROR: Scrape for FDA 510(k) Premarket Notifications failed at {datetime.today()} as URL: {main_URL}, error = Scrape URL not found!'
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)
        return

    lastScrapeDate = scrapeURL.lastScrapeDate

    if lastScrapeDate == None:
        #scrape everything
        main_URL = 'https://www.accessdata.fda.gov/premarket/ftparea/pmn96cur.zip'
        firstTime = True
    else:
        #scrape only current month
        firstTime = False        
        main_URL = 'https://www.accessdata.fda.gov/premarket/ftparea/pmnlstmn.zip'


    if firstTime:
        fda_510k_text_files, status_msg = get_main_page(main_URL, downloadDir)

        if fda_510k_text_files is None:
            logText = f'Scrape for FDA 510(k) Premarket Notifications failed at {datetime.today()} as URL: {main_URL}, error = {status_msg}'
            scrape_url_append_log(mysql_driver, scrapeURLId, logText)
            return

        logText = f'Zipfile data captured at {datetime.today()}'
        logList.append(logText)
        scrape_url_append_log(mysql_driver, scrapeURLId, logText)

        scrapeDF = scrape_510K_records(config, fda_510k_text_files, mysql_driver, scrapeURLId)

    status = insert_510K_records(config, mysql_driver, scrapeDF, scrapeURLId, scrapeScript)
    
    print("final download  list size:")

    logText = f'Scraping URL: {main_URL} DONE'
    logList.append(logText)
    scrape_url_append_log(mysql_driver, scrapeURLId, logText)    
    print(logText)

    #save_log_data_to_db(logList, mysql_driver)
    print('Scaped URL:' + main_URL)


if __name__ == '__main__':
    try:
        props = None
        #configure the logging level
        remaining_args = configure_logging_from_argv(default_level='INFO')

        docIdsList = []
        if len(remaining_args) > 1:
            n = len(remaining_args[0])
            docs = remaining_args[0][1 : n - 1]
            docs = docs.split(" ")
            docIdsList = [int(i) for i in docs]

            if len(docIdsList) > 0:
                scrapeURLId = docIdsList[0]
            else: 
                scrapeURLId = 2

        configs = parseCredentialFile("/app/tlp_config.json")

        if configs:
            run(configs, scrapeURLId)

    except Exception as e:
        print('The EXCEPTION >>>>>>>>>>>>>> ')
        print(traceback.format_exc())
        traceback.print_exc()
        print(e)
