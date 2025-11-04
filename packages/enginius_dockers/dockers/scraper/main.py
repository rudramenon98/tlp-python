import json
import logging
import os
import sys
import time
import traceback
from enum import Enum
from typing import Dict, List, Optional

import faiss
import numpy as np
import requests
from database import CONFIG_DIR
from database.entity.ScriptsProperty import parseCredentialFile
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from scraping import (
    CFR_FAA,
    EASA,
    EU_MDCG,
    EU_MDR,
    FDA_CFR,
    NIST_RMF,
    AC_web,
    Additional_510k,
    AUS_Guidance,
    AUS_Regulation,
    Canada_Regulation,
    EU_MDR_Ammendment,
    FDA_Guidance,
)
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

# --- FastAPI App ---
app = FastAPI(title="Scraper Service API")
# --- Logging Configuration ---
# Configure root logger to output to stdout/stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# --- Logging ---
logger = logging.getLogger(__name__)


# --- Database Configuration ---
DB_CONFIG = parseCredentialFile(str(CONFIG_DIR / "dev_test_tlp_config.json"))

if DB_CONFIG is None:
    logger.error("Failed to load database configuration")
    raise RuntimeError("Database configuration not loaded")


# --- Scraper Enum ---
class Scraper(str, Enum):
    AC_WEB = "AC_web"
    ADDITIONAL_510K = "Additional_510k"
    AUS_GUIDANCE = "AUS_Guidance"
    AUS_REGULATION = "AUS_Regulation"
    CANADA_REGULATION = "Canada_Regulation"
    CFR_FAA = "CFR_FAA"
    EASA = "EASA"
    EU_MDCG = "EU_MDCG"
    EU_MDR_AMMENDMENT = "EU_MDR_Ammendment"
    EU_MDR = "EU_MDR"
    FDA_CFR = "FDA_CFR"
    FDA_GUIDANCE = "FDA_Guidance"
    NIST_RMF = "NIST_RMF"


# Map enum values to module objects
SCRAPER_MODULES = {
    Scraper.AC_WEB: AC_web,
    Scraper.ADDITIONAL_510K: Additional_510k,
    Scraper.AUS_GUIDANCE: AUS_Guidance,
    Scraper.AUS_REGULATION: AUS_Regulation,
    Scraper.CANADA_REGULATION: Canada_Regulation,
    Scraper.CFR_FAA: CFR_FAA,
    Scraper.EASA: EASA,
    Scraper.EU_MDCG: EU_MDCG,
    Scraper.EU_MDR_AMMENDMENT: EU_MDR_Ammendment,
    Scraper.EU_MDR: EU_MDR,
    Scraper.FDA_CFR: FDA_CFR,
    Scraper.FDA_GUIDANCE: FDA_Guidance,
    Scraper.NIST_RMF: NIST_RMF,
}


# --- Pydantic Models ---
class RunScraperRequest(BaseModel):
    scraper: Scraper
    scrapeURLId: Optional[int] = None


class RunScraperResponse(BaseModel):
    success: bool
    message: str


# --- Endpoints ---
@app.get("/scrapers", response_model=List[str])
async def list_scrapers():
    """Returns list of available scrapers defined in the enum."""
    return [scraper.value for scraper in Scraper]


@app.post("/run", response_model=RunScraperResponse)
async def run_scraper(request: RunScraperRequest):
    """
    Executes a scraper's run() method.

    Args:
        request: Contains scraper name and optional scrapeURLId

    Returns:
        Success status and message
    """
    try:
        scraper_module = SCRAPER_MODULES.get(request.scraper)
        if scraper_module is None:
            raise HTTPException(
                status_code=404, message=f"Scraper {request.scraper.value} not found"
            )

        logger.info(f"Running scraper: {request.scraper.value}")

        # Other scrapers require scrapeURLId
        if request.scrapeURLId is None:
            raise HTTPException(
                status_code=400,
                message=f"scrapeURLId is required for scraper {request.scraper.value}",
            )
        logger.info(
            f"Running {request.scraper.value} with scrapeURLId: {request.scrapeURLId}"
        )
        scraper_module.run(DB_CONFIG, request.scrapeURLId)

        return RunScraperResponse(
            success=True,
            message=f"Scraper {request.scraper.value} completed successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running scraper {request.scraper.value}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            message=f"Error running scraper {request.scraper.value}: {str(e)}",
        )


# === Config ===
DIM = 768
INDEX_TYPE = "ivfpq_map"

INDEX_MODE = os.environ.get("INDEX_MODE", "static")
# INDEX_PATH = f'/uploads/Enginius/test/index/enginius_{INDEX_MODE}.index'
INDEX_PATH = f"/index/enginius_{INDEX_MODE}.index"
print(f"Using Index file: {INDEX_PATH}")
if "dynamic" in INDEX_MODE.lower():
    INDEX_TYPE = "flat_map"

# INDEX_PATH = '/index/enginius_hnsw_ivfpq.index'
# INDEX_PATH = '/uploads/Enginius/test/index/enginius.index'
# INDEX_PATH = '/uploads/Enginius/test/index/enginius_hnsw_ivfpq.index'

ENCODER_SERVER = os.environ.get("ENCODER_SERVER", "http://ollama:11434/api/embed")
ENCODER_MODEL = os.environ.get("ENCODER_MODEL", "sbi11.1:q8_0")
print(f"Using Encoder Docker: {ENCODER_SERVER}")
print(f"Using Encoder Model: {ENCODER_MODEL}")

# === State ===
index = None

# --- In-memory DB config cache ---
cached_db_config = None


class DBConfig(BaseModel):
    host: str
    port: int
    username: str
    password: str
    database: str


def sbi_sentence_embedding(encoder_url, paraTextList):
    global ENCODER_MODEL

    headers = {"Content-Type": "application/json"}
    payload = {"model": ENCODER_MODEL, "input": paraTextList, "num_ctx": 256}

    try:
        response = requests.post(
            encoder_url, headers=headers, json=payload, timeout=120
        )

        response.raise_for_status()
        data = response.json()

        print("Model used:", data.get("model", "Unknown"))
        embeddings = data.get("embeddings", [])

        # print(f"\nEmbedding for input {len(embeddings)}")

        return embeddings

    except requests.exceptions.RequestException as e:
        print("Error while making the request:", e)
        return None


# === Utilities ===
def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def create_index(index_type: str, dim: int) -> faiss.Index:
    use_gpu = faiss.get_num_gpus() > 0
    if index_type == "flat":
        idx = faiss.IndexFlatIP(dim)
    elif index_type == "flat_map":
        flat_idx = faiss.IndexFlatIP(dim)
        idx = faiss.IndexIDMap2(flat_idx)
        idx.verbose = True
    elif index_type == "pq":
        idx = faiss.IndexPQ(dim, 16, 8)
    elif index_type == "ivfpq":
        nlist = 4096  # number of inverted lists (coarse quantizer)
        pq_m = 96  # must divide d (768 / 96 = 8)
        nbits = 8  # bits per subquantizer
        # --- Create coarse quantizer ---
        quantizer = faiss.IndexFlatL2(dim)  # flat index for coarse quantization

        # --- Build IVFPQ index ---
        idx = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, nbits)
    elif index_type == "ivfpq_map":

        pq_m = 16  # must divide d (768 / 16 = 48)
        nlist = 32
        nbits = 8  # bits per subquantizer

        # --- Create coarse quantizer ---
        quantizer = faiss.IndexFlatIP(dim)  # flat index for coarse quantization

        # --- Build IVFPQ index ---
        idx_ivfpq = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, nbits)

        # --- Attach a IDMap ---
        idx = faiss.IndexIDMap2(idx_ivfpq)
        idx.verbose = True

    elif index_type == "hnsw":
        idx = faiss.IndexHNSWFlat(dim, 32)
        idx.hnsw.efSearch = 64
        idx.hnsw.efConstruction = 64
    elif index_type == "hnsw_pq":
        # also set M so that the vectors and links both use 128 bytes per
        # entry (total 256 bytes)
        # idx = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, 32)
        idx = faiss.IndexHNSWPQ(dim, 16, 8, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 200  # Good default for high recall
        idx.hnsw.efSearch = 500
    else:
        raise ValueError("Invalid index type")
    return faiss.index_cpu_to_all_gpus(idx) if use_gpu else idx


def save_index():
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(
        faiss.index_gpu_to_cpu(index) if faiss.get_num_gpus() > 0 else index, INDEX_PATH
    )


def load_index(index_type: str = INDEX_TYPE):

    global index

    if os.path.exists(INDEX_PATH):
        idx = faiss.read_index(INDEX_PATH)
        index = faiss.index_cpu_to_all_gpus(idx) if faiss.get_num_gpus() > 0 else idx

    else:
        index = create_index(index_type, DIM)
        if not "dynamic" in INDEX_MODE.lower():
            pass


# === Load index on startup
load_index()


# === Models
class AddRequest(BaseModel):
    texts: List[str]
    ids: List[int]


class DeleteRequest(BaseModel):
    ids: List[int]


class UpdateRequest(BaseModel):
    updates: Dict[int, str]


class ReloadRequest(BaseModel):
    index_type: Optional[str] = None


class RemoveRequest(BaseModel):
    ids: List[int]


class ReindexRequest(BaseModel):
    index_type: Optional[str] = "flat_map"  # default
    train_mode: Optional[str] = "random"  # "sample" or "full"


# === Routes


@app.post("/add")
async def add(request: AddRequest):
    global index, ENCODER_SERVER, current_id

    try:
        if len(request.ids) != len(request.texts):
            raise HTTPException(
                status_code=400, detail="'ids' and 'vectors' length mismatch"
            )

        print("Length of Ids and texts match")

        embeddings = sbi_sentence_embedding(ENCODER_SERVER, request.texts)
        embeddings = np.array(embeddings).astype("float32")
        norm = normalize(embeddings)

        print("texts embedded and normalized ")

        if not index.is_trained:
            raise HTTPException(status_code=400, detail="Faiss index is not trained")
            # index.train(norm)

        print("added to index with ids")
        ids = np.array(request.ids, dtype="int64")  # Faiss expects int64
        index.add_with_ids(norm, ids)
        added_ids = request.ids

        save_index()

        return {"added": added_ids}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def process_form(form_data):
    # seq_0 can be either a single string or a list of strings
    #    raw_seq_0 = form_data.getlist("seq_0")
    #    seq_0: Union[str, List[str]] = (
    #        raw_seq_0[0] if len(raw_seq_0) == 1 else raw_seq_0
    #    )
    # Extract and parse stringified list
    seq_0_raw = form_data.get("seq_0")
    try:
        seq_0 = json.loads(seq_0_raw)
        if not isinstance(seq_0, list) or not all(
            isinstance(item, str) for item in seq_0
        ):
            return {"error": "seq_0 must be a list of strings"}
    except Exception:
        return {"error": "Invalid JSON format for seq_0"}

    # seq_1 should be int
    seq_1_str = form_data.get("seq_1")
    try:
        seq_1 = int(seq_1_str)
    except (TypeError, ValueError):
        return {"error": "Invalid seq_1 value"}

    return {"queries": seq_0, "num_results": seq_1}


@app.post("/search")
async def search(request: Request):

    if index.ntotal == 0:
        return {"results": []}

    print(f"Request = {request.headers['content-type']}")

    if "json" in request.headers["content-type"]:
        body = await request.json()
    elif "x-www-form-urlencoded" in request.headers["content-type"]:
        form_data = await request.form()
        body = process_form(form_data)
    else:
        body = await request.body()
        print(f"body = {body} type = {type(body)}")

    print(f"Request = {request}")

    if "queries" in body and "num_results" in body:
        queries = body["queries"]
        top_k = int(body["num_results"])
    elif "seq_0" in body and "seq_1" in body:
        seq_0 = body["seq_0"]
        queries = [seq_0] if isinstance(seq_0, str) else seq_0
        top_k = int(body["seq_1"])
    else:
        return {"error": "Invalid input format"}

    print(f"queries = {queries}")
    print(f"num_results = {top_k}")
    print(f"Using Encoder Docker: {ENCODER_SERVER}")

    embeddings = sbi_sentence_embedding(ENCODER_SERVER, queries)
    if not embeddings or len(embeddings) == 0:
        return {"error": "Encoding Error"}

    print("Embedding done")
    embeddings = np.array(embeddings).astype("float32")

    norm = normalize(embeddings)

    if top_k > 500:
        top_k = 500

        # Set nprobe if applicable
    if hasattr(index, "nprobe"):
        index.nprobe = 64
    elif hasattr(index, "index") and hasattr(index.index, "nprobe"):
        index.index.nprobe = 64

    t0 = time.time()
    scores, indices = index.search(norm, top_k)

    # --- Deduplicate with max-score logic ---
    result_map = {}

    for i in range(len(queries)):
        for idx, score in zip(indices[i], scores[i]):
            if idx == -1:
                continue
            # Handle mapping depending on static/dynamic index
            real_id = idx
            final_score = (
                1 - score / 2
            )  # Convert from inner product to distance-like score

            # Keep only best score for this index
            if real_id not in result_map or final_score > result_map[real_id]:
                result_map[real_id] = final_score

    # --- Sort results by score descending ---
    sorted_results = sorted(
        [
            {"index": int(idx), "score": float(score)}
            for idx, score in result_map.items()
        ],
        key=lambda x: -x["score"],
    )

    # --- Return top_k ---
    final_results = sorted_results[:top_k]
    t1 = time.time()
    print(f"time taken for search = {t1-t0}")
    # print([final_results])
    return [final_results]


def remove_vectors(ids_to_remove: List[int]) -> Dict:
    global index

    try:
        ids = np.array(ids_to_remove, dtype="int64")
        selector = faiss.IDSelectorBatch(ids)
        removed_count = index.remove_ids(selector)

        save_index()
        return {"status": "removed", "count": removed_count}
    except Exception as e:
        return {"status": "FAIL", "reason": f"Failed to remove IDs: {str(e)}"}


@app.post("/delete")
async def delete(request: DeleteRequest):
    return remove_vectors(request.ids)


@app.post("/update")
async def update(request: UpdateRequest):
    global index, ENCODER_SERVER

    if "dynamic" in INDEX_MODE.lower():

        try:
            vectors = []
            ids = []
            for idx, new_text in request.updates.items():
                emb = sbi_sentence_embedding(ENCODER_SERVER, [new_text])
                vectors.append(emb)
                ids.append(idx)

            id_array = np.array(ids, dtype="int64")
            vector_array = np.vstack(vectors)

            # Remove existing IDs (if any)
            ids2update = np.array(ids, dtype="int64")
            if ids2update:
                selector = faiss.IDSelectorBatch(ids2update)
                index.remove_ids(selector)

            # Add new vectors
            index.add_with_ids(vector_array, id_array)
            return {"status": "updated", "count": len(ids)}

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"FAISS indexing error: {e}"
            ) from e
    else:
        raise HTTPException(
            status_code=500, detail="Updating not supported for static index"
        )


@app.post("/reload")
async def reload(request: ReloadRequest):
    load_index(request.index_type or INDEX_TYPE)
    return {"status": "reloaded", "count": index.ntotal}


@app.post("/remove")
async def remove(request: RemoveRequest):
    return remove_vectors(request.ids)


@app.post("/config")
async def set_config(config: DBConfig):
    global cached_db_config
    cached_db_config = config
    return {"status": "config cached"}


def load_chunk(offset: int, limit: int, db_url: str):

    engine = create_engine(db_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    VECTOR_COLUMNS = [f"vec_{i}" for i in range(768)]
    KEY_COLUMN = "paragraphID"

    if "dynamic" in INDEX_MODE.lower():
        TABLE_NAME = "draft_repository_table"
    else:
        TABLE_NAME = "repository"

    VECTOR_DIM = 768
    VECTOR_COLUMNS = [f"vec_{i}" for i in range(VECTOR_DIM)]

    session = SessionLocal()
    try:
        query = text(
            f"""
            SELECT {KEY_COLUMN}, {', '.join(VECTOR_COLUMNS)}
            FROM {TABLE_NAME}
            WHERE embedding = 1
            ORDER BY {KEY_COLUMN}
            LIMIT :limit OFFSET :offset
            """
        )
        rows = session.execute(query, {"limit": limit, "offset": offset}).fetchall()
        ids = [row[0] for row in rows]
        vectors = np.array([row[1:] for row in rows], dtype="float32")
        return ids, vectors
    finally:
        session.close()
        engine.dispose()


@app.post("/reindex")
async def reindex(request: ReindexRequest):
    global index

    try:
        if DB_CONFIG is not None:
            # extract all database credentials
            cred = DB_CONFIG.databaseConfig.__dict__
            cred["database"]
            db_url = URL.create(
                drivername="mysql+mysqlconnector",
                username=cred["username"],
                password=cred["password"],
                host=cred["host"],
                port=int(cred["port"]),
                database=cred["database"],
            )
        else:
            raise HTTPException(
                status_code=500, detail="Error: Cannot obtain database credentials"
            )

        # Create temp engine to count rows
        engine = create_engine(db_url)
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()

        if "dynamic" in INDEX_MODE.lower():
            total_rows = session.execute(
                text("SELECT COUNT(*) FROM draft_repository_table WHERE embedding = 1")
            ).scalar()
        else:
            total_rows = session.execute(
                text("SELECT COUNT(*) FROM repository WHERE embedding = 1")
            ).scalar()
        session.close()
        engine.dispose()

        if total_rows == 0:
            raise HTTPException(status_code=400, detail="No vectors found in DB.")

        index_type = request.index_type.lower()
        index = create_index(index_type, DIM)

        CHUNK_SIZE = 25000
        offsets = list(range(0, total_rows, CHUNK_SIZE))

        print(
            f"Total rows: {total_rows}, chunk size: {CHUNK_SIZE}, total chunks: {len(offsets)}"
        )

        # === Step 1: Optional training ===
        print("Getting all vectors from DB...")
        all_vectors = []
        all_ids = []
        for offset in offsets:
            ids, vecs = load_chunk(offset, CHUNK_SIZE, db_url)
            all_vectors.append(vecs)
            all_ids.extend(ids)
        norm_vectors = normalize(np.vstack(all_vectors))
        print(f"Got {len(all_ids)} vectors from DB...")

        if not index.is_trained:
            print("Training index...")
            index.train(norm_vectors)
            print("Index trained.")

        print("Adding vectors to index")
        ids = np.array(all_ids, dtype="int64")
        index.add_with_ids(norm_vectors, ids)
        print(f"Added {len(ids)} vectors to index.")

        print(f"Final count in index : {index.ntotal}")
        save_index()

        return {"status": "reindexed", "index_type": index_type, "count": index.ntotal}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/add2")
async def add2(request: AddRequest):
    global index, ENCODER_SERVER, current_id

    try:
        if not request.texts:  # if texts is None or empty
            """
            if config is None:
                raise HTTPException(status_code=500, detail="Cannot obtain DB credentials")
            """
            if DB_CONFIG is None:
                if cached_db_config is None:
                    raise HTTPException(
                        status_code=500, detail="Cannot obtain DB credentials"
                    )
                cred = cached_db_config.dict()
            else:
                cred = DB_CONFIG.databaseConfig.__dict__
            # cred = config.databaseConfig.__dict__

            db_url = URL.create(
                drivername="mysql+mysqlconnector",
                username=cred["username"],
                password=cred["password"],
                host=cred["host"],
                port=int(cred["port"]),
                database=cred["database"],
            )
            engine = create_engine(db_url)
            SessionLocal = sessionmaker(bind=engine)
            session = SessionLocal()
            try:
                # Fetch texts by ids
                query_text = text(
                    "SELECT paragraphID, text FROM draft_repository_table WHERE paragraphID IN :ids"
                )
                rows = session.execute(
                    query_text, {"ids": tuple(request.ids)}
                ).fetchall()
                if not rows or len(rows) < len(request.ids):
                    raise HTTPException(
                        status_code=404, detail="Some IDs not found in DB"
                    )
                id_to_text = {row[0]: row[1] for row in rows}
                texts = [id_to_text[id_] for id_ in request.ids]
            finally:
                session.close()
                engine.dispose()
        else:
            texts = request.texts

        if len(request.ids) != len(texts):
            raise HTTPException(
                status_code=400, detail="'ids' and 'texts' length mismatch"
            )

        # Embedding texts
        embeddings = sbi_sentence_embedding(ENCODER_SERVER, texts)
        embeddings = np.array(embeddings).astype("float32")
        norm = normalize(embeddings)

        if not index.is_trained:
            raise HTTPException(status_code=400, detail="Faiss index is not trained")

        # Add vectors to faiss index
        ids = np.array(request.ids, dtype="int64")
        index.add_with_ids(norm, ids)
        added_ids = request.ids

        # If texts were fetched from DB, update vectors in DB
        if not request.texts:
            vector_columns = [f"vec_{i}" for i in range(norm.shape[1])]
            session = SessionLocal()
            try:
                norm = (128 * norm).astype(np.int8)
                for idx, vector in zip(request.ids, norm):
                    # Build update statement dynamically for vector columns
                    update_stmt = text(
                        f"UPDATE draft_repository_table SET "
                        + ", ".join([f"{col} = :{col}" for col in vector_columns])
                        + " WHERE paragraphID = :pid"
                    )
                    params = {f"vec_{i}": int(vector[i]) for i in range(len(vector))}
                    params["pid"] = idx
                    session.execute(update_stmt, params)
                session.commit()
            except Exception as e:
                session.rollback()
                raise HTTPException(
                    status_code=500, detail=f"Failed to update vectors in DB: {e}"
                ) from e
            finally:
                session.close()
                engine.dispose()

        save_index()

        return {"added": added_ids}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/update2")
async def update2(request: UpdateRequest):
    global index, ENCODER_SERVER

    if "dynamic" in INDEX_MODE.lower():
        try:
            vectors = []
            ids = []

            # Setup DB session
            """
            if DB_CONFIG is None:
                raise HTTPException(status_code=500, detail="Cannot obtain DB credentials")
            """
            
            if DB_CONFIG is None:
                if cached_db_config is None:
                    raise HTTPException(
                        status_code=500, detail="Cannot obtain DB credentials"
                    )
                cred = cached_db_config.dict()
            else:
                cred = DB_CONFIG.databaseConfig.__dict__
            # cred = config.databaseConfig.__dict__
            db_url = URL.create(
                drivername="mysql+mysqlconnector",
                username=cred["username"],
                password=cred["password"],
                host=cred["host"],
                port=int(cred["port"]),
                database=cred["database"],
            )
            engine = create_engine(db_url)
            SessionLocal = sessionmaker(bind=engine)
            session = SessionLocal()

            try:
                print("/update2:")
                print(f"Payload: {request.updates.items()}")
                for idx, new_text in request.updates.items():
                    if (
                        not new_text or len(new_text) == 0
                    ):  # Text missing, fetch from DB
                        query_text = text(
                            "SELECT text FROM draft_repository_table WHERE paragraphID = :pid"
                        )
                        row = session.execute(query_text, {"pid": idx}).fetchone()
                        if row is None:
                            raise HTTPException(
                                status_code=404, detail=f"ID {idx} not found in DB"
                            )
                        new_text = row[0]

                    emb = sbi_sentence_embedding(ENCODER_SERVER, [new_text])
                    vectors.append(emb[0])
                    ids.append(idx)

                ids_array = np.array(ids, dtype="int64")
                vectors_array = np.vstack(vectors)
                norm_vectors = normalize(vectors_array)

                # Remove old vectors in FAISS index
                if ids_array.size > 0:
                    selector = faiss.IDSelectorBatch(ids_array)
                    index.remove_ids(selector)

                # Add new vectors to FAISS index
                index.add_with_ids(norm_vectors, ids_array)

                # Update vectors in DB
                vector_columns = [f"vec_{i}" for i in range(norm_vectors.shape[1])]
                norm_vectors = (128 * norm_vectors).astype(np.int8)
                for idx_, vector in zip(ids, norm_vectors):
                    update_stmt = text(
                        "UPDATE draft_repository_table SET "
                        + ", ".join([f"{col} = :{col}" for col in vector_columns])
                        + " WHERE paragraphID = :pid"
                    )
                    params = {f"vec_{i}": int(vector[i]) for i in range(len(vector))}
                    params["pid"] = idx_
                    session.execute(update_stmt, params)

                session.commit()

            except Exception as e:
                session.rollback()
                raise HTTPException(
                    status_code=500, detail=f"DB update error: {e}"
                ) from e
            finally:
                session.close()
                engine.dispose()

            return {"status": "updated", "count": len(ids)}

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"FAISS or DB processing error: {e}"
            ) from e

    else:
        raise HTTPException(
            status_code=500, detail="Updating not supported for static index"
        ) from e
