from sqlalchemy import BigInteger, Column, Date, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ScrapScript(Base):
    __tablename__ = "scrapescripts"

    scrapeScriptID = Column("scrapeScriptID", BigInteger, primary_key=True)
    title = Column("title", String)
    description = Column("description", String)
    filename = Column("filename", String)
    #    folder = Column("folder", String)
    createdDate = Column("createdDate", Date)
    modifiedDate = Column("modifiedDate", Date)
    documentTypeID = Column("documentTypeID", Integer)
    defaultParsingScriptID = Column("defaultParsingScriptID", Integer)
