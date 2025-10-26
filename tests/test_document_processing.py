import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.document_store import get_document_store

def test_document_store_initialization():
    doc_store = get_document_store()
    assert doc_store is not None
    assert hasattr(doc_store, 'is_ready')

def test_document_store_singleton():
    doc_store1 = get_document_store()
    doc_store2 = get_document_store()
    assert doc_store1 is doc_store2
