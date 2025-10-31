from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    number = Column("number", String)
    documentId = Column("documentID", BigInteger, primary_key=True)
    title = Column("title", String, nullable=False)
    description = Column("description", String)
    url = Column("url", String, nullable=False)
    documentType = Column("documentType", Integer, nullable=False)
    documentStatus = Column("documentStatus", Integer, nullable=False)
    activeDate = Column("activeDate", Date, nullable=False)
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
    summary = Column("summary", String)
    sourceProject = Column("sourceProject", BigInteger)


class DynamicPrivateDocument(Document):
    __tablename__ = "dynamicPrivateDocuments"
    documentId = Column(
        "documentID", BigInteger, ForeignKey("documents.documentID"), primary_key=True
    )


class StaticPrivateDocument(Document):
    __tablename__ = "staticPrivateDocuments"
    documentId = Column(
        "documentID", BigInteger, ForeignKey("documents.documentID"), primary_key=True
    )


class StaticPublicDocument(Document):
    __tablename__ = "staticPublicDocuments"
    documentId = Column(
        "documentID", BigInteger, ForeignKey("documents.documentID"), primary_key=True
    )
