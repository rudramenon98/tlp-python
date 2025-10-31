import io
import logging
import os
import tempfile
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from DocSummarizer import DocumentSummarizer
from app.entity.Document import DynamicPrivateDocument, PrivateStaticDocument, PublicStaticDocument
import requests
from app.entity.Repository import DynamicPrivateRepository, PrivateStaticRepository, PublicStaticRepository, Repository
from app.utils.MySQLFactory import MySQLDriver
from app import configs
from app.document_service import (
    find_document_by_id,
    get_paragraph_vectors_to_be_indexed,
    insert_repository_bulk2,
    set_document_as_parsed,
    set_document_as_embedded,
    set_document_parsed_details,
    set_document_summary,
)
from app.entity.ScriptsProperty import parseCredentialFile
from enginius_parser.inference.ai_pdf_parser import parse_pdf_ai
from fastapi import FastAPI, Form, HTTPException, Request
from joblib import load
from pydantic import BaseModel
from pathlib import Path
# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s")
logger = logging.getLogger(__name__)

# --- Database Configuration ---
# The line `config_path = Path(configs.__file__)` is creating a `Path` object using the `__file__`
# attribute of the `configs` module.
config_path = Path(configs.__file__)
DB_CONFIG = parseCredentialFile(str(config_path.parent / "tlp_config.json"))

# --- Load model ---
#MODEL_PATH = "/models/model.joblib"
#MODEL_PATH = '/uploads/Enginius/test/models/AIPDFParserModel/RandomForestClassifier_modeltest_Production_082920231.1.sav'
# MODEL_PATH = "/uploads/Enginius/test/models/AIPDFParserModel/RandomForestClassifier_modeltest_Production_082920231.1.sav"
MODEL_PATH = "/models/RandomForestClassifier.sav"

# try:
#     model = load(MODEL_PATH)
#     logger.info("Model loaded successfully.")
# except Exception as e:
#     logger.error(f"Model load failed: {e}")
#     raise

# --- Initialize Database Connection ---
try:
    mysql_driver = MySQLDriver(cred=DB_CONFIG.databaseConfig.__dict__)
    logger.info("Database connection initialized successfully.")
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    raise


# --- Pydantic Response Model ---
class PredictionOutput(BaseModel):
    prediction: Dict[str, float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application...")
    try:
        app.state.model = load(MODEL_PATH)
        logger.info("Model loaded successfully in lifespan.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model load failed: {e}")
    yield
    logger.info("Shutting down application...")


# --- Create FastAPI App with Lifespan ---
app = FastAPI(title="PDF Inference API using Lifespan", lifespan=lifespan)


@app.post("/summarize", response_model=bool)
async def summarize(request: Request, doc_ids: List[int] = Form(...)):
    logger.info(f"Summarizing for doc_ids: {doc_ids}")
    try:
        success = True
        failed_docs = []
        logger.info(f"Summarizing for doc_ids: {doc_ids}")
        for doc_id in doc_ids:
            document = find_document_by_id(mysql_driver, doc_id)
            if document is None:
                logger.error(f"Document not found for ID: {doc_id}")
                failed_docs.append({
                    "doc_id": doc_id,
                    "error": "Document not found"
                })
                continue
            paragraphs = get_paragraph_vectors_to_be_indexed(
                mysql_driver, document)
            if not paragraphs:
                logger.error("No paragraphs found for the selected document.")
                success = False
                continue

            # Convert paragraphs to the expected format for summarization
            parsed_content = []
            for para in paragraphs:
                parsed_content.append([(getattr(para, 'data',
                                                ''), getattr(para, 'Type', 0),
                                        getattr(para, 'Page_No', 0))])

            genai_endpoint = f"{DB_CONFIG.GenAIDocker}/api/chat"
            if parsed_content:
                summarizer = DocumentSummarizer(
                    llm_endpoint=genai_endpoint,
                    max_chunk_size=8000,
                    max_paragraph_words=128,
                )
                summary_results = summarizer.summarize_document(
                    parsed_content, save_summaries=False)
                set_document_summary(mysql_driver, document,
                                     summary_results['final_summary'])

        return success

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        logger.error(f"Summarization failed: {traceback.format_exc()}")
        return False


@app.post("/predict", response_model=List[Dict[str, Any]])
async def predict(request: Request,
                  doc_ids: List[int] = Form(...),
                  pdf_type: str = Form(...),
                  repo_id: int = Form(...),
                  ):
    if repo_id == 0:
        doc_type = PrivateStaticDocument
        repo_type = PrivateStaticRepository
        repo_name = "PrivateStatic"
    elif repo_id == 1:
        doc_type = PublicStaticDocument
        repo_type = PublicStaticRepository
        repo_name = "PublicStatic"
    elif repo_id == 2:
        doc_type = DynamicPrivateDocument
        repo_type = DynamicPrivateRepository
        repo_name = "PrivateDynamic"
    else:
        raise HTTPException(status_code=400, detail=f"Invalid repo_id: {repo_id}")
    
    logger.info(f"Base Directory: {DB_CONFIG.baseDir}")
    logger.info(f"Download Directory: {DB_CONFIG.downloadDir}")
    try:
        doc_directory = DB_CONFIG.baseDir + '/scrapedDocs/'
        logger.info(
            f"Predicting for doc_ids: {doc_ids} and pdf_type: {pdf_type}")
        document_df = None
        processed_docs = []
        failed_docs = []

        for doc_id in doc_ids:
            try:
                document = find_document_by_id(mysql_driver, doc_id, doc_type)
                if document is None:
                    logger.error(f"Document not found for ID: {doc_id}")
                    failed_docs.append({
                        "doc_id": doc_id,
                        "error": "Document not found"
                    })
                    continue

                pdf_path = str(Path(doc_directory) / document.pdfFileName)
                logger.info(f"Parsing PDF file: {pdf_path}")
                document_df = parse_pdf_ai(pdf_path,
                                           request.app.state.model,
                                           pdf_type=pdf_type)

                repositories: List[repo_type] = []
                for _, row in document_df.iterrows():
                    repository = repo_type(data=row['text'],
                                            documentID=doc_id,
                                            Type=row['FinalClass'],
                                            pageNo=row['PageNumber'],
                                            wordCount=row['no_of_words'])
                    repositories.append(repository)
                insert_repository_bulk2(mysql_driver, repositories)
                set_document_as_parsed(mysql_driver, document)
                parseLogText = "Successfully parsed document: " + document.pdfFileName + " and inserted " + str(
                    len(repositories)) + " paragraphs into the DB"
                set_document_parsed_details(mysql_driver, document,
                                            parseLogText, len(repositories))
                set_document_as_embedded(mysql_driver, document, False)
                processed_docs.append(doc_id)

            except Exception as doc_error:
                logger.error(
                    f"Failed to process document {doc_id}: {doc_error}")
                failed_docs.append({"doc_id": doc_id, "error": str(doc_error)})

        # Return results for all documents
        results = []
        if processed_docs:
            results.append({
                "status":
                "success",
                "processed_docs":
                processed_docs,
                "message":
                f"Successfully processed {len(processed_docs)} document(s)"
            })
        if failed_docs:
            results.append({
                "status":
                "partial_failure",
                "failed_docs":
                failed_docs,
                "message":
                f"Failed to process {len(failed_docs)} document(s)"
            })

        if not results:
            results.append({
                "status": "error",
                "message": "No documents were processed"
            })

        return results

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.error(f"Prediction failed: {traceback.format_exc()}")
        return [{
            "status": "error",
            "message": str(e),
            "error_type": "system_error"
        }]
