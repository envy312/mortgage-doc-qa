from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_list_documents_empty():
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    assert "total_documents" in response.json()

def test_query_without_documents():
    response = client.post(
        "/api/v1/query",
        json={"query": "What is the loan amount?"}
    )
    assert response.status_code in [200, 400]

def test_upload_invalid_file():
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.txt", b"not a pdf", "text/plain")}
    )
    assert response.status_code == 400