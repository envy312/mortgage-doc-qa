from sqlalchemy import Column, String, Integer, DateTime, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    
    doc_id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    upload_date = Column(DateTime, default=datetime.now)
    total_pages = Column(Integer)
    documents_found = Column(Integer)
    total_chunks = Column(Integer)
    document_types = Column(String)
    processing_time = Column(String)
    ocr_pages = Column(Integer)
    avg_confidence = Column(String)
    status = Column(String, default='processing')

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/mortgage_docs.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
