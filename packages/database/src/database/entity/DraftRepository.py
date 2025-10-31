from sqlalchemy import BigInteger, Boolean, Column, DateTime, Integer, String
from sqlalchemy.dialects.mysql import INTEGER, TINYINT
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
UnsignedInteger = BigInteger()
UnsignedInteger = UnsignedInteger.with_variant(INTEGER(unsigned=True), "mysql")

UnsignedTinyInt = Integer()
UnsignedTinyInt = UnsignedTinyInt.with_variant(TINYINT(unsigned=True), "mysql")


class DraftRepository(Base):
    __tablename__ = "draft_repository_table"

    paragraphID = Column("paragraphID", BigInteger, primary_key=True)
    data = Column("data", String)
    wordCount = Column("wordCount", Integer)
    #    instruction = Column("instruction", String)
    documentID = Column("documentID", BigInteger, nullable=False)
    #    source_section_field_form_id = Column("source_section_field_form_id", String)
    chatid = Column("chatid", String)
    embedding = Column("embedding", Boolean)
    createdDate = Column("createdDate", DateTime)
    modifiedDate = Column("modifiedDate", DateTime)
    #    source_section_widget_template_id = Column("source_section_widget_template_id", String)
    Type = Column("Type", Integer)
    pageNo = Column("pageNo", Integer)
    vec_0 = Column("vec_0", Integer)

    tokenCount = Column("tokenCount", Integer)
