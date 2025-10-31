import traceback
from typing import List

from database.entity.Document import Document
from database.entity.Repository import Repository
from database.entity.ScrapeURL import ScrapeURL
from database.entity.ScrapScript import ScrapScript
from database.entity.Vector import Vector
from database.entity.VocabularyPartV2 import VocabularyPartV2
from database.utils.MySQLFactory import MySQLDriver
from sqlalchemy import and_, func, text


def Error_Handler(fn):
    def Inner_Function(*args, **kwargs):
        try:
            ret = fn(*args, **kwargs)
            return ret
        except Exception:
            print(f"Exception in {fn.__name__}")
            traceback.print_exc()
            raise

    return Inner_Function


def find_match_obj(row, result):
    for x in result:
        if x.url == row.url:
            return x

    return None


def get_repository_class_for_document(document: Document, repository_base_cls):
    """
    Automatically returns the corresponding Repository class for a given Document instance
    based on naming convention: <Name>Document -> <Name>Repository
    """
    # Get the class name of the document
    doc_cls_name = document.__class__.__name__

    # Determine expected repository class name
    repo_cls_name = doc_cls_name.replace("Document", "Repository")

    # Search among repository subclasses
    repo_subclasses = repository_base_cls.__subclasses__() + [repository_base_cls]

    for repo_cls in repo_subclasses:
        if repo_cls.__name__ == repo_cls_name:
            return repo_cls

    raise ValueError(f"No repository class found for {doc_cls_name}")


@Error_Handler
def check_documents_needs_to_download(
    mysql_driver, document_type, document_list, doc_class=Document
):
    session = mysql_driver.get_session()
    result = (
        session.query(doc_class).filter(doc_class.documentType == document_type).all()
    )
    download_list = []
    delete_list = []
    if result:
        for row in document_list:
            # matched_obj = next(x for x in result if x.url == row.url)
            matched_obj = find_match_obj(row, result)
            if not matched_obj:
                download_list.append(row)
            elif matched_obj.activeDate < row.activeDate:
                print("Latest doc available for download")
                delete_list.append(matched_obj)
                download_list.append(row)
    else:
        download_list.extend(document_list)
    session.close()
    return download_list


@Error_Handler
def find_document_by_url(mysql_driver, URL, doc_class=Document):
    session = mysql_driver.get_session()
    result = session.query(doc_class).filter(doc_class.url == URL).first()
    session.close()
    return result


@Error_Handler
def find_document_by_id(mysql_driver, doc_id, doc_class=Document):
    session = mysql_driver.get_session()
    result = session.query(doc_class).filter(doc_class.documentId == doc_id).first()
    session.close()
    return result

@Error_Handler
def get_all_document_ids(mysql_driver) -> List[int]:
    session = mysql_driver.get_session()
    result = session.query(Document.documentId).all()
    ids = [row[0] for row in result]
    session.close()
    return ids

@Error_Handler
def find_documents_not_scraped_on_date(mysql_driver, currentDate, doc_class=Document):
    session = mysql_driver.get_session()
    # result = session.query(Document) \
    #    .filter(Document.lastScrapeDate < currentDate) \
    #    .filter(Document.documentStatus == 1) \
    #    .all()
    result = (
        session.query(doc_class)
        .filter(
            and_(doc_class.lastScrapeDate < currentDate, doc_class.documentStatus == 1)
        )
        .all()
    )
    session.close()
    return result


@Error_Handler
def find_documents_not_scraped_on_date_by_type(
    mysql_driver, currentDate, document_type, doc_class=Document
):
    session = mysql_driver.get_session()
    result = (
        session.query(doc_class)
        .filter(doc_class.documentStatus != 2)
        .filter(doc_class.documentType == document_type)
        .filter(doc_class.lastScrapeDate < currentDate)
        .all()
    )
    session.close()
    return result


def get_document_by_title(mysql_driver: MySQLDriver, title: str) -> Document:
    session = mysql_driver.get_session()
    result = session.query(Document).filter(Document.title == title).first()
    session.close()
    return result


@Error_Handler
def find_documents_by_type(mysql_driver, document_type, doc_class=Document):
    session = mysql_driver.get_session()
    result = (
        session.query(doc_class).filter(doc_class.documentType == document_type).first()
    )
    session.close()
    return result


@Error_Handler
def insert_document(mysql_driver, document):
    session = mysql_driver.get_session()
    session.add(document)
    session.commit()
    session.close()


@Error_Handler
def insert_documents_bulk(mysql_driver, document_list):
    session = mysql_driver.get_session()
    for doc in document_list:
        session.add(doc)
        session.commit()
    session.close()


@Error_Handler
def insert_repository_bulk(mysql_driver, repository_data_list):
    session = mysql_driver.get_session()
    for doc in repository_data_list:
        session.add(doc)
        session.commit()
    session.close()


@Error_Handler
def insert_repository_bulk2(mysql_driver, repository_data_list):
    session = mysql_driver.get_session()
    for doc in repository_data_list:
        session.add(doc)
        session.flush()
    session.commit()
    session.close()


@Error_Handler
def insert_vocabulary_bulk(mysql_driver, vocabulary_data_list):
    session = mysql_driver.get_session()
    for doc in vocabulary_data_list:
        session.add(doc)
        session.commit()
    session.close()


@Error_Handler
def empty_vocabulary(mysql_driver):
    session = mysql_driver.get_session()
    session.query(VocabularyPartV2).delete()
    session.commit()
    session.close()


@Error_Handler
def get_documents_for_embedding(mysql_driver, doc_class=Document):
    session = mysql_driver.get_session()
    result = (
        session.query(doc_class)
        .filter(and_(doc_class.embedded == 0, doc_class.parsed == 1))
        .all()
    )
    session.close()
    return result


@Error_Handler
def get_all_documents(mysql_driver, doc_class=Document):
    session = mysql_driver.get_session()
    result = session.query(doc_class).all()
    session.close()
    return result


@Error_Handler
def get_documents_for_parsing(mysql_driver, doc_class=Document):
    session = mysql_driver.get_session()
    result = (
        session.query(doc_class)
        .filter(and_(doc_class.parsed == 0, doc_class.documentStatus != 2))
        .order_by(doc_class.documentId)
        .all()
    )
    session.close()
    return result


@Error_Handler
def get_documents_for_parsing_by_type(mysql_driver, document_type, doc_class=Document):
    session = mysql_driver.get_session()
    result = (
        session.query(doc_class)
        .filter(and_(doc_class.parsed == 0, doc_class.documentType == document_type))
        .all()
    )
    # result = session.query(Document).filter(Document.documentType == document_type).all()
    session.close()
    return result


@Error_Handler
def get_paragraphs_to_be_embedded(mysql_driver, document: Document, size=100):
    repo_class = get_repository_class_for_document(document, Repository)

    session = mysql_driver.get_session()
    result = (
        session.query(repo_class)
        .filter(
            and_(
                repo_class.documentID == document.documentId,
                repo_class.vector.is_(None),
            )
        )
        .order_by(repo_class.paragraphID)
        .limit(size)
        .all()
    )
    session.close()
    return result


@Error_Handler
def get_all_paragraphs_to_be_embedded(
    mysql_driver, isToFetchAll, size=100, repo_class=Repository
):
    session = mysql_driver.get_session()
    if isToFetchAll:
        result = (
            session.query(repo_class).order_by(repo_class.paragraphID).limit(size).all()
        )
    else:
        result = (
            session.query(repo_class)
            .filter(and_(repo_class.vector.is_(None)))
            .order_by(repo_class.paragraphID)
            .limit(size)
            .all()
        )
    session.close()
    return result


@Error_Handler
def get_paragraphs_to_be_embedded_count(mysql_driver, document: Document):
    repo_class = get_repository_class_for_document(document, Repository)

    session = mysql_driver.get_session()
    total_rows = (
        session.query(func.count(repo_class.paragraphID))
        .filter(
            and_(
                repo_class.documentID == document.documentId, repo_class.vector == None
            )
        )
        .scalar()
    )
    session.close()
    return total_rows


@Error_Handler
def updates_paragraphs(mysql_driver, paragraph_list):
    session = mysql_driver.get_session()
    bulk_update_mappings = []
    for para in paragraph_list:
        bulk_update_mappings.append(
            {"paragraphID": para.paragraphID, "vector": para.vector}
        )
    session.bulk_update_mappings(Repository, bulk_update_mappings)
    session.commit()
    session.close()


@Error_Handler
def set_document_as_embedded(mysql_driver, doc: Document, doc_class=Document):
    session = mysql_driver.get_session()
    session.query(doc_class).filter(doc_class.documentId == doc.documentId).update(
        {"embedded": True}
    )
    session.commit()
    session.close()


@Error_Handler
def set_document_parsed_details(
    mysql_driver, doc: Document, parsingLogText, noOfParagraphs, doc_class=Document
):
    session = mysql_driver.get_session()
    parsingLogText = parsingLogText[:4095]
    session.query(doc_class).filter(doc_class.documentId == doc.documentId).update(
        {"parsingLog": parsingLogText, "noOfParagraphs": noOfParagraphs}
    )
    session.commit()
    session.close()


@Error_Handler
def set_document_embedding_details(
    mysql_driver, doc: Document, embeddingText, doc_class=Document
):
    session = mysql_driver.get_session()
    session.query(doc_class).filter(doc_class.documentId == doc.documentId).update(
        {"embeddingLog": embeddingText}
    )
    session.commit()
    session.close()


@Error_Handler
def set_document_summary(mysql_driver, doc: Document, summaryText, doc_class=Document):
    session = mysql_driver.get_session()
    session.query(doc_class).filter(doc_class.documentId == doc.documentId).update(
        {"summary": summaryText}
    )
    session.commit()
    session.close()


@Error_Handler
def set_document_as_parsed(mysql_driver, doc: Document, doc_class=Document):
    session = mysql_driver.get_session()
    session.query(doc_class).filter(doc_class.documentId == doc.documentId).update(
        {"parsed": True}
    )
    session.commit()
    session.close()


@Error_Handler
def delete_repository_data_by_doc_id(mysql_driver, docId, repo_class=Repository):
    session = mysql_driver.get_session()
    session.query(repo_class).filter(repo_class.documentID == docId).delete()
    session.commit()
    session.close()


@Error_Handler
def update_documents(mysql_driver, documents_list, doc_class=Document):
    session = mysql_driver.get_session()

    for doc in documents_list:
        session.query(doc_class).filter(doc_class.documentId == doc.documentId).update(
            {"lastScrapeDate": doc.lastScrapeDate}
        )
        # update({"lastScrapeDate": doc.lastScrapeDate, "scrapingLog": doc.scrapingLog})

        session.commit()
    session.close()


@Error_Handler
def update_documents2(mysql_driver, documents_list, doc_class=Document):
    session = mysql_driver.get_session()

    for doc in documents_list:
        session.query(doc_class).filter(doc_class.documentId == doc.documentId).update(
            {
                "lastScrapeDate": doc.lastScrapeDate,
                "pdfFileName": doc.pdfFileName,
                "fileSize": doc.fileSize,
                "documentStatus": doc.documentStatus,
                "parsed": doc.parsed,
                "documentStatus": 1,
            }
        )
        session.commit()
    session.close()


@Error_Handler
def cancel_documents(mysql_driver, documents_list, doc_class=Document):
    session = mysql_driver.get_session()

    for doc in documents_list:
        session.query(doc_class).filter(doc_class.documentId == doc.documentId).update(
            {
                "documentStatus": doc.documentStatus,
                "inactiveDate": doc.inactiveDate,
                "modifiedDate": doc.modifiedDate,
            }
        )

        session.commit()
    session.close()


@Error_Handler
def set_document_status_as_cancelled(mysql_driver, doc, doc_class=Document):
    session = mysql_driver.get_session()
    session.query(doc_class).filter(doc_class.documentId == doc.documentId).update(
        {
            "documentStatus": doc.documentStatus,
            "inactiveDate": doc.inactiveDate,
            "modifiedDate": doc.modifiedDate,
        }
    )

    session.commit()
    session.close()


@Error_Handler
def get_scrap_script_by_file_name(mysql_driver, fileName):
    session = mysql_driver.get_session()
    result = session.query(ScrapScript).filter(ScrapScript.filename == fileName).first()
    return result


@Error_Handler
def get_paragraph_vectors_to_be_indexed(mysql_driver, doc):
    repo_class = get_repository_class_for_document(doc, Repository)

    session = mysql_driver.get_session()
    result = (
        session.query(repo_class)
        .filter(repo_class.documentID == doc.documentId)
        .order_by(repo_class.paragraphID)
        .all()
    )
    session.close()
    return result


def get_paragraph_vectors_to_be_indexed_batched(mysql_driver, doc, batch_size=10):
    """
    Generator that yields paragraphs in batches (lists) of size `batch_size`
    from the Repository table for a given document.
    """
    repo_class = get_repository_class_for_document(doc, Repository)

    session = mysql_driver.get_session()
    try:
        query = (
            session.query(repo_class)
            #            .filter(Repository.documentID == doc.documentId)
            .filter(
                and_(repo_class.documentID == doc.documentId, repo_class.embedding == 0)
            ).order_by(repo_class.paragraphID)
        )
        # and_(Repository.documentID == document.documentId, Repository.vector == None)
        offset = 0
        while True:
            batch = query.limit(batch_size).offset(offset).all()
            if not batch:
                break
            yield batch
            offset += batch_size
    finally:
        session.close()


@Error_Handler
def get_all_paragraph_vectors_to_be_indexed(mysql_driver, repo_class=Repository):
    session = mysql_driver.get_session()
    result = session.query(repo_class).order_by(repo_class.paragraphID).all()
    session.close()
    return result


@Error_Handler
def insert_vectors_bulk(mysql_driver, vectorsSQL_list):
    session = mysql_driver.get_session()
    for vecSQL in vectorsSQL_list:
        session.execute(text(vecSQL))
        session.commit()
    session.close()


@Error_Handler
def get_all_vectors_to_be_indexed(mysql_driver, sql_stmt):
    session = mysql_driver.get_session()
    # result = session.execute(text('select * from MED_DEV_STAGE.vectors n;'))
    result = session.execute(text(sql_stmt))
    session.close()
    return result


@Error_Handler
def get_max_scannOrderNo(mysql_driver):
    session = mysql_driver.get_session()

    result = session.query(func.max(Vector.scannOrderNo)).scalar()
    session.close()
    return result


@Error_Handler
def set_SCaNNOrderNo_in_repository(mysql_driver, para: Repository, orderNo):
    session = mysql_driver.get_session()
    session.query(Repository).filter(Repository.paragraphID == para.paragraphID).update(
        {"scannOrderNo": orderNo}
    )
    session.commit()
    session.close()


@Error_Handler
def insert_vectors_bulk2(mysql_driver, vectorsSQL_list, paragraphs_list):
    session = mysql_driver.get_session()

    # insert embedding vectors into 'vectors' table
    for vecSQL in vectorsSQL_list:
        session.execute(text(vecSQL))
        session.commit()

    # update scannorderno in 'respository_v2' table
    #    bulk_update_mappings = []
    #    for para in paragraphs_list:
    #        bulk_update_mappings.append({
    #            "paragraphID": para[0],
    #            "scannOrderNo": para[1]
    #        })
    #    session.bulk_update_mappings(Repository, bulk_update_mappings)
    for para in paragraphs_list:
        session.query(Repository).filter(Repository.paragraphID == para[0]).update(
            {"scannOrderNo": para[1]}
        )
        session.commit()

    # session.commit()
    session.close()


@Error_Handler
def update_vectors_bulk(main_mysql_driver, vectorsSQL_list):
    main_session = main_mysql_driver.get_session()

    # update embedding vectors into repository table's vectors columns
    for vecSQL in vectorsSQL_list:
        main_session.execute(text(vecSQL))
        main_session.commit()
    main_session.close()


@Error_Handler
def insert_vectors_bulk3(main_mysql_driver, vectorsSQL_list, paragraphs_list):
    main_session = main_mysql_driver.get_session()

    # insert embedding vectors into 'vectors' table
    for vecSQL in vectorsSQL_list:
        main_session.execute(text(vecSQL))
        main_session.commit()
    #    main_session.close()

    #    main_session = main_mysql_driver.get_session()
    for para in paragraphs_list:
        main_session.query(Repository).filter(Repository.paragraphID == para[0]).update(
            {"scannOrderNo": para[1]}
        )
        main_session.commit()

    # session.commit()
    main_session.close()


@Error_Handler
def get_scrape_script_by_scraperUrlId(mysql_driver, scrapeURLid):
    session = mysql_driver.get_session()
    scrapeURL = session.query(ScrapeURL).filter(ScrapeURL.URLID == scrapeURLid).first()
    if scrapeURL is not None:
        print(
            f"ScrapeURL row found for {scrapeURLid} ScrapeScriptID: {scrapeURL.scrapeScriptID}"
        )
        result = (
            session.query(ScrapScript)
            .filter(ScrapScript.scrapeScriptID == scrapeURL.scrapeScriptID)
            .first()
        )

        return result
    else:
        print(f"ScrapeURL row not found for {scrapeURLid}")
        return None


@Error_Handler
def insert_documents_bulk2(mysql_driver, document_list):
    documentID_list = []
    session = mysql_driver.get_session()
    for doc in document_list:
        if doc is None:
            continue
        print(f"Inserting document into DB {doc.number}")
        session.add(doc)
        session.commit()
        documentID_list.append(doc.documentId)
    # session.commit()
    session.close()
    return documentID_list


@Error_Handler
def get_parsing_script_by_document_type_name(mysql_driver, sql_stmt: str):
    session = mysql_driver.get_session()
    result = session.execute(text(sql_stmt))
    session.close()
    return result


@Error_Handler
def get_scrapeurl_by_scrapeUrlId(mysql_driver, scrapeURLid):
    session = mysql_driver.get_session()
    scrapeURL = session.query(ScrapeURL).filter(ScrapeURL.URLID == scrapeURLid).first()
    session.close()
    if scrapeURL is not None:
        return scrapeURL
    else:
        print(f"ScrapeURL row not found for {scrapeURLid}")
        return None


@Error_Handler
def find_document_by_number(mysql_driver, number, doc_class=Document):
    session = mysql_driver.get_session()
    result = session.query(doc_class).filter(doc_class.number == number).first()
    session.close()
    return result
