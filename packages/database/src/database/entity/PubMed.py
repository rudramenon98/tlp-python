from sqlalchemy import BigInteger, Boolean, Column, Date, Integer, SmallInteger, String
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

UnsignedBigInteger = BigInteger()
UnsignedBigInteger = UnsignedBigInteger.with_variant(BIGINT(unsigned=True), "mysql")


class PubMed(Base):
    __tablename__ = "pubmed"

    #    pmid = Column('pmid', Integer, nullable=False, primary_key=True)
    pmid = Column("pmid", UnsignedBigInteger, nullable=False, primary_key=True)
    pmidversion = Column("pmidversion", SmallInteger, nullable=False)
    title = Column("title", String(1024), nullable=True)
    abstract = Column("abstractText", String(8192), nullable=False)
    titleParagraphID = Column("titleParagraphID", Integer, nullable=True)
    abstractParagraphID = Column("abstractParagraphID", Integer, nullable=True)
    authors = Column("authors", String(2048), nullable=False)
    url = Column("url", String(255), nullable=False)
    deleted = Column("deleted", Boolean, default=False)
    createdDate = Column("createdDate", Date)
    modifiedDate = Column("modifiedDate", Date)
