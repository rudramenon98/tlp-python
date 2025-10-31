import logging
import traceback

from database.entity.ScrapeURL import ScrapeURL

log = logging.getLogger(__name__)


def update_scrape_url_set_log_value(mysql_driver, scrape_url_id, log):
    try:
        session = mysql_driver.get_session()
        session.query(ScrapeURL).filter(ScrapeURL.URLID == scrape_url_id).update(
            {"scrapingLog": log}
        )
        session.commit()
        session.close()
    except Exception as e:
        log.debug("Error while saving logs into ScrapURL %s", e)
        traceback.print_exc()


def scrape_url_append_log(mysql_driver, scrape_url_id, logText):
    try:
        session = mysql_driver.get_session()
        scrape_url = (
            session.query(ScrapeURL).filter(ScrapeURL.URLID == scrape_url_id).first()
        )

        scrapeLog = scrape_url.scrapingLog
        if scrapeLog is None:
            scrapeLog = logText + "\n"
        else:
            scrapeLog += logText + "\n"
        session.query(ScrapeURL).filter(ScrapeURL.URLID == scrape_url_id).update(
            {"scrapingLog": scrapeLog}
        )
        session.commit()
        session.close()
    except Exception as e:
        log.debug("Error while appending Scraping log into ScrapURL %s", e)
        traceback.print_exc()
