from sqlalchemy import BigInteger, Boolean, Column, Date, DateTime, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DraftDocument(Base):
    __tablename__ = "draft_documents"

    documentID = Column("documentID", BigInteger, primary_key=True)
    title = Column("title", String)
    description = Column("description", String)
    documentType = Column("documentType", Integer, nullable=False)
    documentStatus = Column("documentStatus", Integer)
    activeDate = Column("activeDate", Date)
    inactiveDate = Column("inactiveDate", Date)
    sourceFileName = Column("sourceFileName", String)
    pdfFileName = Column("pdfFileName", String)
    parsed = Column("parsed", Boolean)
    embedded = Column("embedded", Boolean)
    parsingScriptID = Column("parsingScriptID", Integer)
    createdDate = Column("createdDate", DateTime)
    modifiedDate = Column("modifiedDate", DateTime)

    fileSize = Column("fileSize", BigInteger)
    parsingLog = Column("parsingLog", String)
    embeddingLog = Column("embeddingLog", String)

    noOfParagraphs = Column("noOfParagraphs", Integer)

    lastScrapeDate = Column("lastScrapeDate", DateTime)
    number = Column("number", String)
    summary = Column("summary", String)

    sourceProject = Column("sourceProject", BigInteger)
    sourceTemplateId = Column("sourceTemplateId", Integer)

    backgroundDocumentFile = Column("backgroundDocumentFile", String)
    documentStructureList = Column("documentStructureList", String)
    createdDate = Column("createdDate", Date, nullable=False)
    modifiedDate = Column("modifiedDate", Date, nullable=False)
