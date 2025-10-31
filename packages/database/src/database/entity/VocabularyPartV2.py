from sqlalchemy import BigInteger, Column, Float, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class VocabularyPartV2(Base):
    __tablename__ = "vocabularypart_v2"

    tokenID = Column("tokenID", BigInteger, primary_key=True)
    word = Column("word", String, nullable=True)
    frequencyPercent = Column("frequencyPercent", Float)
    frequencyCount = Column("frequencyCount", BigInteger)
    vector = Column("vector", String)
