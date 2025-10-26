from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import sys
import uuid
from datetime import datetime
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.document_store import get_document_store
from database.models import get_db, Document, create_tables

app = FastAPI(
    title="Mortgage Document Q&A API",
    description="RAG-based document analysis for mortgage applications",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

create_tables()
doc_store = get_document_store()

class QueryRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    top_k: int = 5
    auto_route: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float
    filter_used: str
    chunks_used: int

@app.get("/")
def root():
    return {
        "message": "Mortgage Document Q&A API",
        "status": "running",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "document_store_ready": doc_store.is_ready,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a mortgage document (PDF)
    
    - Supports multi-document PDFs
    - Handles both native and scanned documents
    - Returns processing statistics
    """
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    doc_id = str(uuid.uuid4())
    
    os.makedirs("data/uploads", exist_ok=True)
    file_path = f"data/uploads/{doc_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        success, stats = doc_store.process_pdf(file_path, filename=file.filename)
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Processing failed: {stats.get('error')}")
        
        db = next(get_db())
        doc_record = Document(
            doc_id=doc_id,
            filename=file.filename,
            file_path=file_path,
            total_pages=stats['total_pages'],
            documents_found=stats['documents_found'],
            total_chunks=stats['total_chunks'],
            document_types=",".join(stats['document_types']),
            processing_time=stats['processing_time'],
            ocr_pages=stats['ocr_pages'],
            avg_confidence=stats['avg_confidence'],
            status='processed'
        )
        db.add(doc_record)
        db.commit()
        db.refresh(doc_record)
        db.close()
        
        return {
            "status": "success",
            "doc_id": doc_id,
            "filename": file.filename,
            "statistics": stats
        }
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Query processed documents
    
    - Uses three-stage retrieval (dense + sparse + reranking)
    - Supports query expansion for vague queries
    - Returns answers with source attribution
    """
    
    if not doc_store.is_ready:
        raise HTTPException(
            status_code=400,
            detail="No documents processed. Please upload documents first."
        )
    
    try:
        result = doc_store.query(
            request.query,
            auto_route=request.auto_route,
            k=request.top_k
        )
        
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources'],
            confidence=result['confidence'],
            filter_used=result['filter_used'],
            chunks_used=result['chunks_used']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/documents")
def list_documents():
    """
    List all processed documents with metadata
    """
    
    db = next(get_db())
    documents = db.query(Document).all()
    db.close()
    
    return {
        "total_documents": len(documents),
        "documents": [
            {
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "upload_date": doc.upload_date.isoformat(),
                "total_pages": doc.total_pages,
                "document_types": doc.document_types.split(","),
                "status": doc.status,
                "processing_time": doc.processing_time
            }
            for doc in documents
        ]
    }

@app.get("/api/v1/documents/{doc_id}")
def get_document_info(doc_id: str):
    """
    Get detailed information about a specific document
    """
    
    db = next(get_db())
    doc = db.query(Document).filter(Document.doc_id == doc_id).first()
    db.close()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "upload_date": doc.upload_date.isoformat(),
        "total_pages": doc.total_pages,
        "documents_found": doc.documents_found,
        "total_chunks": doc.total_chunks,
        "document_types": doc.document_types.split(","),
        "processing_time": doc.processing_time,
        "ocr_pages": doc.ocr_pages,
        "avg_confidence": doc.avg_confidence,
        "status": doc.status
    }

@app.delete("/api/v1/documents/{doc_id}")
def delete_document(doc_id: str):
    """
    Delete a document and its associated data
    """
    
    db = next(get_db())
    doc = db.query(Document).filter(Document.doc_id == doc_id).first()
    
    if not doc:
        db.close()
        raise HTTPException(status_code=404, detail="Document not found")
    
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)
    
    db.delete(doc)
    db.commit()
    db.close()
    
    return {"status": "success", "message": f"Document {doc_id} deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
