import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Dictionary Service", version="1.0.0")


@app.get("/")
async def root():
    return {"message": "Dictionary Service is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/process")
async def process_document():
    # Add your document processing logic here
    return {"message": "Document processing endpoint"}


class ExtractTermsRequest(BaseModel):
    document_ids: Optional[List[int]] = None
    document_types: Optional[List[int]] = None


@app.post("/extract-terms")
async def extract_terms(body: ExtractTermsRequest):
    # Placeholder implementation - replace with real extraction logic
    return {
        "message": "Terms extraction endpoint",
        "document_ids": body.document_ids or [],
        "document_types": body.document_types or [],
    }


@app.get("/info")
async def get_info():
    return {
        "service": "Dictionary Service",
        "version": "1.0.0",
        "status": "running",
        "python_path": os.environ.get("PYTHONPATH", "not set"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
