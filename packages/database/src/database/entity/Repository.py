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


class DynamicPrivateRepository(Repository):
    __tablename__ = "dynamicPrivateRepository"

    paragraphID = Column(
        "paragraphID",
        UnsignedInteger,
        ForeignKey("repository.paragraphID"),
        primary_key=True,
    )


class StaticPrivateRepository(Repository):
    __tablename__ = "staticPrivateRepository"

    paragraphID = Column(
        "paragraphID",
        UnsignedInteger,
        ForeignKey("repository.paragraphID"),
        primary_key=True,
    )


class StaticPublicRepository(Repository):
    __tablename__ = "staticPublicRepository"

    paragraphID = Column(
        "paragraphID",
        UnsignedInteger,
        ForeignKey("repository.paragraphID"),
        primary_key=True,
    )
