import os
import json
import time
from typing import List, Dict, Union, Optional
import requests

from fastapi import FastAPI, Request, HTTPException

from pydantic import BaseModel


import faiss

import numpy as np

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import URL

from ScriptsProperty import ScriptsConfig, parseCredentialFile
from database.entity.Document import DraftDocument, PrivateDocument, PublicDocument, getDocumentClass
import requests
from database.entity.Repository import DraftRepository, PrivateRepository, PublicRepository, getRepositoryClass
from database.utils.MySQLFactory import MySQLDriver
from database import configs
from database.entity.ScriptsProperty import parseCredentialFile
app = FastAPI()

# === Config ===
DIM = 768
INDEX_TYPE = 'hnsw_pq_map'

INDEX_MODE = os.environ.get("INDEX_MODE", "public")
#INDEX_PATH = f'/uploads/Enginius/test/index/enginius_{INDEX_MODE}.index'
INDEX_PATH = f'/index/enginius_{INDEX_MODE}.index'
print(f"Using Index file: {INDEX_PATH}")
if 'draft' in INDEX_MODE.lower():
    INDEX_TYPE = 'flat_map'
elif 'private' in INDEX_MODE.lower():
    INDEX_TYPE = 'ivfpq_map'
elif 'public' in INDEX_MODE.lower():
    INDEX_TYPE = 'hnsw_pq_map'

#INDEX_PATH = '/index/enginius_hnsw_ivfpq.index'
#INDEX_PATH = '/uploads/Enginius/test/index/enginius.index'
#INDEX_PATH = '/uploads/Enginius/test/index/enginius_hnsw_ivfpq.index'

ENCODER_MODEL = os.environ.get("ENCODER_MODEL", "sbi11.2v1")
ENCODER_SERVER = os.environ.get("ENCODER_SERVER", f"http://localhost:9885/v2/models/{ENCODER_MODEL}/infer")

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
    payload = {"model": ENCODER_MODEL, "input": paraTextList, "num_ctx":256}

    try:
        response = requests.post(encoder_url, headers=headers, json=payload, timeout =120)

        response.raise_for_status()
        data = response.json()

        print("Model used:", data.get("model", "Unknown"))
        embeddings = data.get("embeddings", [])

        #print(f"\nEmbedding for input {len(embeddings)}")

        return embeddings

    except requests.exceptions.RequestException as e:
        print("Error while making the request:", e)
        return None

def sbi_sentence_embedding_triton(
    triton_url,
    texts
):
    """
    Get text embeddings from a Triton Inference Server model using the HTTP binary format.

    Args:
        texts (list of str): List of input text strings.
        model_name (str): Triton model name.
        triton_url (str): URL for Triton server HTTP endpoint.
        max_seq_len (int): Maximum allowed sequence length.
        seq_len_multiplier (int): Multiplier for sequence length based on max token count per text.

    Returns:
        np.ndarray: Embeddings array with shape as defined by the model's output.
    """
,
    model_name=ENCODER_MODEL,
    max_seq_len=256,
    seq_len_multiplier=2,
    #triton_url="http://localhost:9885",
    #triton_url = "http://vllm-encoder2:8000/v1/embeddings"
    #triton_url = config.EncoderDocker

    if not texts:
        raise ValueError("Input `texts` list cannot be empty.")
    if not all(isinstance(t, str) for t in texts):
        raise TypeError("All items in `texts` must be strings.")

    t0 = time.time()

    batch_size = len(texts)

    # Calculate sequence length based on max tokens, capped by max_seq_len
    max_len = max(len(t.split()) for t in texts)
    seq_len = min(max_len * seq_len_multiplier, max_seq_len)

    # Prepare binary-encoded BYTES tensor for text input
    def encode_triton_bytes(strings):
        buf = bytearray()
        for s in strings:
            b = s.encode("utf-8")
            buf += len(b).to_bytes(4, byteorder="little")
            buf += b
        return bytes(buf)

    text_tensor_bytes = encode_triton_bytes(texts)
    seq_tensor_bytes = np.full((batch_size, 1), seq_len, dtype=np.int32).tobytes()

    inputs = [
        {
            "name": "text",
            "shape": [batch_size, 1],
            "datatype": "BYTES",
            "parameters": {"binary_data_size": len(text_tensor_bytes)},
        },
        {
            "name": "seq_length",
            "shape": [batch_size, 1],
            "datatype": "INT32",
            "parameters": {"binary_data_size": len(seq_tensor_bytes)},
        },
    ]

    outputs = [{"name": "embedding", "parameters": {"binary_data": True}}]

    json_body = {"inputs": inputs, "outputs": outputs}
    json_bytes = json.dumps(json_body).encode("utf-8")

    infer_url = f"{triton_url}/v2/models/{model_name}/infer"
    headers = {"Inference-Header-Content-Length": str(len(json_bytes))}

    # Combine JSON header and binary input tensors
    body = json_bytes + text_tensor_bytes + seq_tensor_bytes

    try:
        resp = requests.post(infer_url, headers=headers, data=body)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to get inference from Triton server: {e}")

    # Parse response header and binary output
    header_length_str = resp.headers.get("Inference-Header-Content-Length")
    if header_length_str is None:
        raise RuntimeError("Missing Inference-Header-Content-Length in response headers.")
    header_length = int(header_length_str)

    json_header = resp.content[:header_length]
    binary_data = resp.content[header_length:]

    result = json.loads(json_header)
    output_info = result["outputs"][0]

    # Validate output shape
    shape = output_info.get("shape")
    if shape is None:
        raise RuntimeError("Output shape not found in Triton response.")

    dtype = np.float32  # Adjust if your model outputs a different datatype
    num_elements = int(np.prod(shape))

    embeddings = np.frombuffer(binary_data, dtype=dtype, count=num_elements)
    embeddings = embeddings.reshape(shape)

    t1 = time.time()
    print(f"Inference time: {t1 - t0:.3f}s for batch of {batch_size}")
    
    #normalize the embeddings
    norm_embeddings = (128*embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]).astype(int)

    return norm_embeddings.tolist()


# === Utilities ===
def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def create_index(index_type: str, dim: int) -> faiss.Index:
    use_gpu = faiss.get_num_gpus() > 0
    if index_type == 'flat':
        idx = faiss.IndexFlatIP(dim)
    elif index_type == 'flat_map':
        flat_idx = faiss.IndexFlatIP(dim)
        idx = faiss.IndexIDMap2(flat_idx)
        idx.verbose = True
    elif index_type == 'pq':
        idx = faiss.IndexPQ(dim, 16, 8)
    elif index_type == 'ivfpq':
        nlist = 4096             # number of inverted lists (coarse quantizer)
        pq_m = 96                # must divide d (768 / 96 = 8)
        nbits = 8                # bits per subquantizer
        # --- Create coarse quantizer ---
        quantizer = faiss.IndexFlatL2(dim)  # flat index for coarse quantization

        # --- Build IVFPQ index ---
        idx = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, nbits)
    elif index_type == 'ivfpq_map':
        
        pq_m = 16                # must divide d (768 / 16 = 48)                
        nlist = 32
        nbits = 8                # bits per subquantizer

        # --- Create coarse quantizer ---
        quantizer = faiss.IndexFlatIP(dim)  # flat index for coarse quantization

        # --- Build IVFPQ index ---
        idx_ivfpq = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, nbits)

        # --- Attach a IDMap ---
        idx = faiss.IndexIDMap2(idx_ivfpq)
        idx.verbose = True

    elif index_type == 'hnsw':
        idx = faiss.IndexHNSWFlat(dim, 32)
        idx.hnsw.efSearch = 64
        idx.hnsw.efConstruction = 64
    elif index_type == 'hnsw_pq':
        # also set M so that the vectors and links both use 128 bytes per
        # entry (total 256 bytes)
        #idx = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, 32)
        idx = faiss.IndexHNSWPQ(dim, 16, 8, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 200  # Good default for high recall
        idx.hnsw.efSearch = 500
    elif index_type == 'hnsw_pq_map':
        # also set M so that the vectors and links both use 128 bytes per
        # entry (total 256 bytes)
        d = 768        # Dimension (length) of vectors.
        idx = faiss.index_factory(d, "IDMap2,IVF65536_HNSW32,PQ64", faiss.METRIC_IP)
        idx.verbose = True
        # nprobe: number of IVF lists to search (higher = better recall, slower)
        idx.nprobe = 10
        # efSearch: HNSW search parameter (higher = better recall, slower)
        # This parameter needs to be set on the coarse quantizer itself
        idx.quantizer.hnsw.efSearch = 64

    else:
        raise ValueError("Invalid index type")
    return faiss.index_cpu_to_all_gpus(idx) if use_gpu else idx


def save_index():
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(faiss.index_gpu_to_cpu(index) if faiss.get_num_gpus() > 0 else index, INDEX_PATH)

def load_index(index_type: str = INDEX_TYPE):

    global index

    if os.path.exists(INDEX_PATH) :
        idx = faiss.read_index(INDEX_PATH)
        index = faiss.index_cpu_to_all_gpus(idx) if faiss.get_num_gpus() > 0 else idx

    else:
        index = create_index(index_type, DIM)
        if not 'dynamic' in INDEX_MODE.lower():
            active_ids = set()
            id_map = dict()
            current_id = 0

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
            raise HTTPException(status_code=400, detail="'ids' and 'vectors' length mismatch")

        print('Length of Ids and texts match')

        #embeddings = sbi_sentence_embedding(ENCODER_SERVER, request.texts)
        embeddings = sbi_sentence_embedding_triton(ENCODER_SERVER, request.texts)
        
        embeddings = np.array(embeddings).astype("float32")
        norm = normalize(embeddings)

        print('texts embedded and normalized ')

        if not index.is_trained:
            raise HTTPException(status_code=400, detail="Faiss index is not trained")
            #index.train(norm)

        print('added to index with ids')
        ids = np.array(request.ids, dtype='int64')  # Faiss expects int64
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
        if not isinstance(seq_0, list) or not all(isinstance(item, str) for item in seq_0):
            return {"error": "seq_0 must be a list of strings"}
    except Exception:
        return {"error": "Invalid JSON format for seq_0"}

    # seq_1 should be int
    seq_1_str = form_data.get("seq_1")
    try:
        seq_1 = int(seq_1_str)
    except (TypeError, ValueError):
        return {"error": "Invalid seq_1 value"}

    return {
        "queries": seq_0,
        "num_results": seq_1
    }

@app.post("/search")
async def search(request: Request):

    if index.ntotal == 0:
        return {"results": []}

    print(f"Request = {request.headers['content-type']}")

    if 'json' in request.headers['content-type']:
        body = await request.json()
    elif 'x-www-form-urlencoded' in request.headers['content-type']:
        form_data = await request.form()
        body  = process_form(form_data)
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

    #embeddings = sbi_sentence_embedding(ENCODER_SERVER, queries)
    embeddings = sbi_sentence_embedding_triton(ENCODER_SERVER, queries)

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
            final_score = 1 - score / 2  # Convert from inner product to distance-like score

            # Keep only best score for this index
            if real_id not in result_map or final_score > result_map[real_id]:
                result_map[real_id] = final_score

    # --- Sort results by score descending ---
    sorted_results = sorted(
        [{"index": int(idx), "score": float(score)} for idx, score in result_map.items()],
        key=lambda x: -x["score"]
    )

    # --- Return top_k ---
    final_results =  sorted_results[:top_k]
    t1 = time.time()
    print(f"time taken for search = {t1-t0}")
    #print([final_results])
    return [final_results]

def remove_vectors(ids_to_remove: List[int]) -> Dict:
    global index

    try:
        ids = np.array(ids_to_remove, dtype='int64')
        selector = faiss.IDSelectorBatch(ids)
        removed_count = index.remove_ids(selector)

        save_index()
        return {
            "status": "removed",
            "count": removed_count
        }
    except Exception as e:
        return {"status": "FAIL", "reason": f"Failed to remove IDs: {str(e)}"}


@app.post("/delete")
async def delete(request: DeleteRequest):
    return remove_vectors(request.ids)

@app.post("/update")
async def update(request: UpdateRequest):
    global index, ENCODER_SERVER

    try:
        vectors = []
        ids = []
        for idx, new_text in request.updates.items():
            #emb = sbi_sentence_embedding(ENCODER_SERVER, [new_text])
            emb = sbi_sentence_embedding_triton(ENCODER_SERVER, [new_text])
            vectors.append(emb)
            ids.append(idx)
    
        id_array = np.array(ids, dtype='int64')
        vector_array = np.vstack(vectors)

        # Remove existing IDs (if any)
        ids2update = np.array(ids, dtype='int64')
        if ids2update:
            selector =faiss.IDSelectorBatch(ids2update)
            index.remove_ids(selector)

        # Add new vectors
        index.add_with_ids(vector_array, id_array)
        return {"status": "updated", "count": len(ids)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS indexing error: {e}") from e

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

    if "draft" in INDEX_MODE.lower():
        TABLE_NAME = "draftRepository"
    elif "private" in INDEX_MODE.lower()::
        TABLE_NAME = "privateRepository"
    elif "public" in INDEX_MODE.lower()::
        TABLE_NAME = "publicRepository"

    VECTOR_DIM = 768

    session = SessionLocal()
    try:
        query = text(
            f"""
            SELECT {KEY_COLUMN}, vector
            FROM {TABLE_NAME}
            WHERE embedding = 1
            ORDER BY {KEY_COLUMN}
            LIMIT :limit OFFSET :offset
            """
        )
        rows = session.execute(query, {"limit": limit, "offset": offset}).fetchall()
        ids = [row[0] for row in rows]

        # Extract vectors from VARBINARY(768) and convert to 2D NumPy array
        vectors = []
        for row in rows:
            # row[1] is a bytes object of length 768
            v = np.frombuffer(row[1], dtype=np.int8).tolist()  # use int8 because stored as bytes
            vectors.append(v)

        # Stack into 2D array: shape = (num_rows, 768)
        #vectors = np.stack(vectors, axis=0)
        return ids, vectors
    finally:
        session.close()
        engine.dispose()

@app.post("/reindex")
async def reindex(request: ReindexRequest):
    global index

    try:

        config = parseCredentialFile('/app/tlp_config.json')
        if config is not None:
            #extract all database credentials
            cred = config.databaseConfig.__dict__
            db_name = cred['database']
            db_url = URL.create(
                drivername="mysql+mysqlconnector",
                username=cred['username'],
                password=cred['password'],
                host=cred['host'],
                port=int(cred['port']),
                database=cred['database'],
            )
        else:
            raise HTTPException(status_code=500, detail="Error: Cannot obtain database credentials")

        # Create temp engine to count rows
        engine = create_engine(db_url)
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()

        if "draft" in INDEX_MODE.lower():
            TABLE_NAME = "draftRepository"
        elif "private" in INDEX_MODE.lower()::
            TABLE_NAME = "privateRepository"
        elif "public" in INDEX_MODE.lower()::
            TABLE_NAME = "publicRepository"

        total_rows = session.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE embedding = 1")).scalar()
        session.close()
        engine.dispose()

        if total_rows == 0:
            raise HTTPException(status_code=400, detail="No vectors found in DB.")

        index_type = request.index_type.lower()
        index = create_index(index_type, DIM)

        CHUNK_SIZE = 25000
        offsets = list(range(0, total_rows, CHUNK_SIZE))

        print(f"Total rows: {total_rows}, chunk size: {CHUNK_SIZE}, total chunks: {len(offsets)}")

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
        ids = np.array(all_ids, dtype='int64')
        index.add_with_ids(norm_vectors, ids)
        print(f"Added {len(ids)} vectors to index.")

        print(f"Final count in index : {index.ntotal}")
        save_index()

        return {
            "status": "reindexed",
            "index_type": index_type,
            "count": index.ntotal
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/add2")
async def add2(request: AddRequest):
    global index, ENCODER_SERVER, current_id

    try:
        if not request.texts:  # if texts is None or empty
            '''
            config = parseCredentialFile('test-tlp_config.json')
            if config is None:
                raise HTTPException(status_code=500, detail="Cannot obtain DB credentials")
            '''
            config = parseCredentialFile('/app/tlp_config.json')
            if config is None:
                if cached_db_config is None:
                    raise HTTPException(status_code=500, detail="Cannot obtain DB credentials")
                cred = cached_db_config.dict()
            else:
                cred = config.databaseConfig.__dict__
            #cred = config.databaseConfig.__dict__

            db_url = URL.create(
                drivername="mysql+mysqlconnector",
                username=cred['username'],
                password=cred['password'],
                host=cred['host'],
                port=int(cred['port']),
                database=cred['database'],
            )
            engine = create_engine(db_url)
            SessionLocal = sessionmaker(bind=engine)
            session = SessionLocal()
            try:
                # Fetch texts by ids
                query_text = text(
                    "SELECT paragraphID, text FROM draft_repository_table WHERE paragraphID IN :ids"
                )
                rows = session.execute(query_text, {"ids": tuple(request.ids)}).fetchall()
                if not rows or len(rows) < len(request.ids):
                    raise HTTPException(status_code=404, detail="Some IDs not found in DB")
                id_to_text = {row[0]: row[1] for row in rows}
                texts = [id_to_text[id_] for id_ in request.ids]
            finally:
                session.close()
                engine.dispose()
        else:
            texts = request.texts

        if len(request.ids) != len(texts):
            raise HTTPException(status_code=400, detail="'ids' and 'texts' length mismatch")

        # Embedding texts
        #embeddings = sbi_sentence_embedding(ENCODER_SERVER, texts)
        embeddings = sbi_sentence_embedding_triton(ENCODER_SERVER, texts)
        embeddings = np.array(embeddings).astype("float32")
        norm = normalize(embeddings)

        if not index.is_trained:
            raise HTTPException(status_code=400, detail="Faiss index is not trained")

        # Add vectors to faiss index
        ids = np.array(request.ids, dtype='int64')
        index.add_with_ids(norm, ids)
        added_ids = request.ids

        # If texts were fetched from DB, update vectors in DB
        if not request.texts:
            vector_columns = [f"vec_{i}" for i in range(norm.shape[1])]
            session = SessionLocal()
            try:
                norm = (128* norm).astype(np.int8)
                for idx, vector in zip(request.ids, norm):
                    # Build update statement dynamically for vector columns
                    update_stmt = text(
                        f"UPDATE draft_repository_table SET " +
                        ", ".join([f"{col} = :{col}" for col in vector_columns]) +
                        " WHERE paragraphID = :pid"
                    )
                    params = {f"vec_{i}": int(vector[i]) for i in range(len(vector))}
                    params["pid"] = idx
                    session.execute(update_stmt, params)
                session.commit()
            except Exception as e:
                session.rollback()
                raise HTTPException(status_code=500, detail=f"Failed to update vectors in DB: {e}") from e
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

    try:
        if "draft" in INDEX_MODE.lower():
            TABLE_NAME = "draftRepository"
        elif "private" in INDEX_MODE.lower()::
            TABLE_NAME = "privateRepository"
        elif "public" in INDEX_MODE.lower()::
            TABLE_NAME = "publicRepository"

        vectors = []
        ids = []

        # Setup DB session
        '''
        config = parseCredentialFile('test-tlp_config.json')
        if config is None:
            raise HTTPException(status_code=500, detail="Cannot obtain DB credentials")
        '''
        config = parseCredentialFile('/app/tlp_config.json')
        if config is None:
            if cached_db_config is None:
                raise HTTPException(status_code=500, detail="Cannot obtain DB credentials")
            cred = cached_db_config.dict()
        else:
            cred = config.databaseConfig.__dict__
        #cred = config.databaseConfig.__dict__
        db_url = URL.create(
            drivername="mysql+mysqlconnector",
            username=cred['username'],
            password=cred['password'],
            host=cred['host'],
            port=int(cred['port']),
            database=cred['database'],
        )
        engine = create_engine(db_url)
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()

        try:
            print("/update2:")
            print(f"Payload: {request.updates.items()}")
            for idx, new_text in request.updates.items():
                if not new_text or len(new_text) == 0:  # Text missing, fetch from DB
                    query_text = text(
                        f"SELECT text FROM {TABLE_NAME} WHERE paragraphID = :pid"
                    )
                    row = session.execute(query_text, {"pid": idx}).fetchone()
                    if row is None:
                        raise HTTPException(status_code=404, detail=f"ID {idx} not found in DB")
                    new_text = row[0]

                #emb = sbi_sentence_embedding(ENCODER_SERVER, [new_text])
                emb = sbi_sentence_embedding_triton(ENCODER_SERVER, [new_text])
                vectors.append(emb[0])
                ids.append(idx)

            ids_array = np.array(ids, dtype='int64')
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

            # Assuming norm_vectors is shape (num_rows, 768) and dtype int8
            norm_vectors = (128 * norm_vectors).astype(np.int8)

            for idx_, vector in zip(ids, norm_vectors):
                # Convert the 1D int8 numpy array to bytes
                vector_bytes = vector.tobytes()

                # Update statement: single placeholder for VARBINARY column
                update_stmt = text(f"""
                    UPDATE {TABLE_NAME} 
                    SET vector = :vector
                    WHERE paragraphID = :pid
                """)

                # Parameters dictionary
                params = {
                    "vector": vector_bytes,
                    "pid": idx_
                }

                # Execute the update
                session.execute(update_stmt, params)

            session.commit()

        except Exception as e:
            session.rollback()
            raise HTTPException(status_code=500, detail=f"DB update error: {e}") from e
        finally:
            session.close()
            engine.dispose()

        return {"status": "updated", "count": len(ids)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS or DB processing error: {e}") from e

