from sqlalchemy import BigInteger, Boolean, Column, Date, DateTime, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Definition(Base):
    __tablename__ = "dictionary"

    id = Column("id", BigInteger, primary_key=True, autoincrement=True)
    term = Column("term", String(255), nullable=False)
    categoryID = Column("categoryID", Integer)
    type = Column("type", Integer)
    sourceId = Column("sourceId", Integer)
    expansionCount = Column("expansionCount", Integer)
    repoFreq = Column("repoFreq", Integer)
    uploadDocName = Column("uploadDocName", String(255))
    tdVocabId = Column("tdVocabId", Integer)
    tsseVocabId = Column("tsseVocabId", Integer)
    tdVocabAddDate = Column("tdVocabAddDate", Date)
    createdDate = Column("createdDate", DateTime)
    modifiedDate = Column("modifiedDate", DateTime)
    approved = Column("approved", Boolean)
