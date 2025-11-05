# 🧠 Vector Indexing API – REST Documentation

This API enables vector embedding, indexing, searching, updating, and managing sentence embeddings using **FAISS**, **FastAPI**, and a sentence transformer backend.

---

## ⚙️ Configuration

| Variable         | Description                             | Default                             |
|------------------|-----------------------------------------|-------------------------------------|
| `INDEX_MODE`     | Static or dynamic indexing              | `static`                            |
| `ENCODER_SERVER` | URL of the embedding server             | `http://ollama:11434/api/embed`     |
| `ENCODER_MODEL`  | Model name for sentence embeddings      | `sbi11.1:q8_0`                       |
| `INDEX_PATH`     | FAISS index file path                   | `/index/enginius_{INDEX_MODE}.index` |

---

## 🚀 Endpoints Overview

| Method | Endpoint     | Description                               |
|--------|--------------|-------------------------------------------|
| POST   | `/add`       | Add new text vectors                      |
| POST   | `/add2`      | Add vectors with DB fallback              |
| POST   | `/search`    | Search top K similar vectors              |
| POST   | `/delete`    | Delete vectors by IDs                     |
| POST   | `/remove`    | Alias for delete                          |
| POST   | `/update`    | Update vector for given ID (dynamic only) |
| POST   | `/update2`   | Update vector with DB fallback            |
| POST   | `/reload`    | Reload FAISS index from disk              |
| POST   | `/reindex`   | Recreate index from DB                    |
| POST   | `/config`    | Set DB credentials                        |

---

## 📦 Request Object Entities

### 🔸 AddRequest (`/add`, `/add2`)

| Field   | Type         | Required | Description                                       |
|---------|--------------|----------|---------------------------------------------------|
| `texts` | `List[str]`  | Yes      | List of text strings to be embedded               |
| `ids`   | `List[int]`  | Yes      | List of unique integer IDs for the texts          |

> In `/add`, len(`texts`) == len('ids') > 0.
> In `/add2`, if `texts` is empty, it will fetch texts from the database using the `ids`.

---

### 🔸 DeleteRequest / RemoveRequest (`/delete`, `/remove`)

| Field | Type        | Required | Description                           |
|-------|-------------|----------|---------------------------------------|
| `ids` | `List[int]` | Yes      | IDs of vectors to be removed          |

---

### 🔸 UpdateRequest (`/update`, `/update2`)

| Field     | Type                | Required | Description                                                                 |
|-----------|---------------------|----------|-----------------------------------------------------------------------------|
| `updates` | `Dict[int, str]`    | Yes      | Dictionary mapping ID to updated text. If text is empty (`""`), will fetch from DB in `/update2`. |

---

### 🔸 ReloadRequest (`/reload`)

| Field        | Type             | Required | Description                                           |
|--------------|------------------|----------|-------------------------------------------------------|
| `index_type` | `Optional[str]`  | No       | FAISS index type to reload (e.g., `"ivfpq_map"`)      |

---

### 🔸 ReindexRequest (`/reindex`)

| Field        | Type             | Required | Description                                                        |
|--------------|------------------|----------|--------------------------------------------------------------------|
| `index_type` | `Optional[str]`  | No       | Type of FAISS index to recreate (`flat_map`, `ivfpq_map`)        |
| `train_mode` | `Optional[str]`  | No       | Training mode for the index (`full`, `sample`)           |

> Only 'full' is implemented.

---

### 🔸 ConfigRequest (`/config`)
| Entity          | Field        | Type                     | Description                                    |
|-----------------|--------------|--------------------------|------------------------------------------------|
| `ConfigRequest` | `username`   | string                   | DB username for MySQL connection               |
|                 | `password`   | string                   | DB password                                    |
|                 | `host`       | string                   | DB host address                                |
|                 | `port`       | integer                  | DB port number                                 |
|                 | `database`   | string                   | Database name                                  |



---

### 🔸 `/search` Input Formats

#### JSON Input
| Field         | Type        | Required | Description                          |
|---------------|-------------|----------|--------------------------------------|
| `queries`     | `List[str]` | Yes      | List of input queries                |
| `num_results` | `int`       | Yes      | Number of top results to return (max 500) |

#### Form Data Input (x-www-form-urlencoded)
| Field   | Type              | Required | Description                              |
|---------|-------------------|----------|------------------------------------------|
| `seq_0` | `str` (JSON list) | Yes      | A JSON-encoded string list of query texts |
| `seq_1` | `int`             | Yes      | Number of results to return              |

---

## 🟢 `/add`

Add vectors for the given texts and IDs.

### ✅ Request:
```json
{
  "texts": ["Document 1", "Document 2"],
  "ids": [101, 102]
}
```

### 🔁 Response:
```json
{
  "added": [101, 102]
}
```

---

## 🟢 `/add2`

Same as `/add`, but if `texts` is empty, fetch them from the database using `ids`.

### ✅ Request:
```json
{
  "texts": [],
  "ids": [123, 124]
}
```

### 🔁 Response:
```json
{
  "added": [123, 124]
}
```

---

## 🔍 `/search`

Search top K nearest vectors using input text.

### ✅ JSON Payload:
```json
{
  "queries": ["What is artificial intelligence?"],
  "num_results": 5
}
```

### ✅ Form Data Payload:
```
seq_0 = ["text query here"]
seq_1 = 5
```

### 🔁 Response:
```json
[
  [
    {"index": 101, "score": 0.95},
    {"index": 104, "score": 0.89}
  ]
]
```

---

## ❌ `/delete`

Delete vectors by ID.

### ✅ Request:
```json
{
  "ids": [101, 102]
}
```

### 🔁 Response:
```json
{
  "status": "removed",
  "count": 2
}
```

---

## ❌ `/remove`

Alias for `/delete`.

---

## 🔁 `/update`

Update existing vectors with new text (only for `dynamic` index mode).

### ✅ Request:
```json
{
  "updates": {
    "101": "New text for ID 101",
    "102": "Updated content"
  }
}
```

### 🔁 Response:
```json
{
  "status": "updated",
  "count": 2
}
```

---

## 🔁 `/update2`

Same as `/update`, but fetch text from DB if empty.

### ✅ Request:
```json
{
  "updates": {
    "123": "",
    "124": "Updated document for 124"
  }
}
```

### 🔁 Response:
```json
{
  "status": "updated",
  "count": 2
}
```

---

## 🔄 `/reload`

Reload the FAISS index from disk. Useful after manual file updates or switching index types.

### ✅ Request:
```json
{
  "index_type": "ivfpq_map" // Optional
}
```

### 🔁 Response:
```json
{
  "status": "reloaded",
  "count": 25000
}
```

---

## 🔁 `/reindex`

Rebuild the FAISS index from the vectors stored in the MySQL database.

### ✅ Request:
```json
{
  "index_type": "flat_map",
  "train_mode": "random"
}
```

### 🔁 Response:
```json
{
  "status": "reindexed",
  "index_type": "flat_map",
  "count": 100000
}
```

---

## 🔁 `/config`

Sets and caches MySQL database credentials used by /add2 and /update2 endpoints when the tlp_config.json file is not found..

### ✅ Request:
```json
{
  "username": "string",
  "password": "string",
  "host": "string",
  "port": integer,
  "database": "string"
}
```

### 🔁 Response:
```json
{
  "status": "success",
  "message": "Database config cached"
}
```

---

## 🧠 Embedding API (Internal)

This service sends queries to the embedding server.

### 📝 Request:
```json
POST /api/embed
{
  "model": "sbi11.1:q8_0",
  "input": ["Your sentence here"],
  "num_ctx": 256
}
```

### ✅ Response:
```json
{
  "model": "sbi11.1:q8_0",
  "embeddings": [[0.123, -0.456, ...]]
}
```

---

## 🗃️ Database Schema

### Used tables:

- **Dynamic Index Mode**: `draft_repository_table`
- **Static Index Mode**: `repository`

### Relevant columns:
- `paragraphID`: Primary key (int)
- `text`: Original document (string)
- `embedding`: Flag (bool/int) if vector is stored
- `vec_0` to `vec_767`: Vector components (float)

---

## ⚠️ Notes

- `/update`, `/update2` only work in **dynamic** index mode.
- Top-K for `/search` is capped at **500 results**.
- GPU acceleration is used if available.
- Index is automatically saved after mutation operations.

---

## ❌ Error Responses

All errors follow this format:
```json
{
  "detail": "Error message goes here"
}
```

---


