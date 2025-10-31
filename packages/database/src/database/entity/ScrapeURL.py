from sqlalchemy import BigInteger, Boolean, Column, Date, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ScrapeURL(Base):
    __tablename__ = "scrapeurls"

    URLID = Column("URLID", BigInteger, primary_key=True)
    # rootURL = Column("rootURL", String)
    description = Column("description", String)
    lastScrapeDate = Column("lastScrapeDate", Date)
    scrapeFrequency = Column("scrapeFrequency", BigInteger)
    scrapeScriptID = Column("scrapeScriptID", BigInteger)
    # folder = Column("folder", String)
    active = Column("active", Boolean)
    status = Column("status", String)
    # documentTypeId = Column("documentTypeID", Integer)
    jobStatus = Column("jobStatus", String)
    jobId = Column("jobId", String)
    jobRunnerId = Column("jobRunnerId", String)
    createdDate = Column("createdDate", Date)
    modifiedDate = Column("modifiedDate", Date)
    # parsingScriptID = Column("parsingScriptID", BigInteger)
    scrapingLog = Column("scrapingLog", String)
