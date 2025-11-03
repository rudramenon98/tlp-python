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


class DraftDocument(Document):
    __tablename__ = "draftDocuments"
    documentId = Column(
        "documentID", BigInteger, ForeignKey("documents.documentID"), primary_key=True
    )


class PrivateDocument(Document):
    __tablename__ = "privateDocuments"
    documentId = Column(
        "documentID", BigInteger, ForeignKey("documents.documentID"), primary_key=True
    )


class PublicDocument(Document):
    __tablename__ = "publicDocuments"
    documentId = Column(
        "documentID", BigInteger, ForeignKey("documents.documentID"), primary_key=True
    )


def getDocumentClass(repo_id: int):
    if repo_id == 0:
        doc_class = PublicDocument
    elif repo_id == 1:
        doc_class = PrivateDocument
    elif repo_id == 2:
        doc_class = DraftDocument
    elif repo_id == 3:
        doc_class = Document

    return doc_class
