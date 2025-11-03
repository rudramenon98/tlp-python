from sqlalchemy import BigInteger, Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.dialects.mysql import INTEGER, TINYINT
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
UnsignedInteger = BigInteger()
UnsignedInteger = UnsignedInteger.with_variant(INTEGER(unsigned=True), "mysql")

UnsignedTinyInt = Integer()
UnsignedTinyInt = UnsignedTinyInt.with_variant(TINYINT(unsigned=True), "mysql")


class Repository(Base):
    __tablename__ = "repository"

    paragraphID = Column("paragraphID", UnsignedInteger, primary_key=True)
    data = Column("data", String)
    documentID = Column("documentID", BigInteger, nullable=False)
    # pageNoText = Column("pageNoText", String)
    pageNo = Column("pageNo", Integer)
    embedding = Column("embedding", Boolean)
    #    wordCount = Column("wordCount", UnsignedTinyInt)
    wordCount = Column("wordCount", Integer)
    #    paraNo = Column("paraNo", Integer)
    Type = Column("Type", Integer)
    scannOrderNo = Column("scannOrderNo", UnsignedInteger)


class DraftRepository(Repository):
    __tablename__ = "DraftRepository"

    paragraphID = Column(
        "paragraphID",
        UnsignedInteger,
        ForeignKey("repository.paragraphID"),
        primary_key=True,
    )


class PrivateRepository(Repository):
    __tablename__ = "PrivateRepository"

    paragraphID = Column(
        "paragraphID",
        UnsignedInteger,
        ForeignKey("repository.paragraphID"),
        primary_key=True,
    )


class PublicRepository(Repository):
    __tablename__ = "PublicRepository"

    paragraphID = Column(
        "paragraphID",
        UnsignedInteger,
        ForeignKey("repository.paragraphID"),
        primary_key=True,
    )


def getRepositoryClass(repo_id: int):
    if repo_id == 0:
        repo_class = PublicRepository
    elif repo_id == 1:
        repo_class = PrivateRepository
    elif repo_id == 2:
        repo_class = DraftRepository
    elif repo_id == 3:
        repo_class = Repository

    return repo_class
