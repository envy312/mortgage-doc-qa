import pymupdf as fitz
from PyPDF2 import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime
import hashlib
import warnings
import re
import cv2
from rank_bm25 import BM25Okapi

from groq import Groq
import easyocr
from PIL import Image
import io

from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

import os

warnings.filterwarnings("ignore")


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set. Please set it in your .env file.")

groq_client = Groq(api_key=GROQ_API_KEY)
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

@dataclass
class PageInfo:
    page_num: int
    text: str
    doc_type: Optional[str] = None
    page_in_doc: int = 0
    ocr_used: bool = False
    confidence: float = 1.0

@dataclass
class LogicalDocument:
    doc_id: str
    doc_type: str
    page_start: int
    page_end: int
    text: str
    chunks: List[Dict] = None

@dataclass
class ChunkMetadata:
    chunk_id: str
    doc_id: str
    doc_type: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    embedding: Optional[np.ndarray] = None

def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced

def clean_extracted_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('|', 'I')
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    text = re.sub(r'(\$)\s+(\d)', r'\1\2', text)

    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 2 and not re.match(r'^\d{1,3}$', line):
            cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def extract_text_hybrid(page, page_num: int) -> Tuple[str, bool, float]:
    global ocr_reader
    if 'ocr_reader' not in globals():
        print("Initializing EasyOCR (first time only)...")
        ocr_reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)
    
    native_text = page.get_text().strip()

    if len(native_text) > 50 and len(native_text.split()) > 10:
        cleaned_text = clean_extracted_text(native_text)
        return cleaned_text, False, 1.0

    print(f"   Page {page_num+1}: Using OCR for text extraction...")
    try:
        pix = page.get_pixmap(dpi=300)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        img_array = np.array(img)
        processed_img = preprocess_image_for_ocr(img_array)

        ocr_result = ocr_reader.readtext(processed_img, detail=1, paragraph=True)

        text_parts = []
        confidences = []
        for detection in ocr_result:
            if len(detection) >= 2:
                text_parts.append(detection[1])
                if len(detection) >= 3:
                    confidences.append(detection[2])

        text = "\n".join(text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0.5

        if avg_confidence < 0.7 and len(native_text) > 0:
            text = native_text + "\n" + text

        cleaned_text = clean_extracted_text(text)
        print(f"   Page {page_num+1}: OCR complete (confidence: {avg_confidence:.2%}, {len(cleaned_text)} chars)")

        return cleaned_text, True, avg_confidence

    except Exception as e:
        print(f"   Page {page_num+1}: OCR failed - {e}")
        return clean_extracted_text(native_text) if native_text else "", True, 0.0
    
def classify_mortgage_document(text: str, max_length: int = 3000) -> str:
    text_sample = text[:1500] + "\n...\n" + text[-1500:] if len(text) > 3000 else text

    prompt = f"""Analyze this document and classify it into ONE category:

Categories:
- Employment Contract: Contract of employment, employment agreement, terms of employment
- Resume: CV, professional profile, work history, career summary
- Pay Stub: Salary statement, wage slip, payslip, earnings statement
- Loan Estimate: Loan estimate, fee worksheet, closing disclosure
- Loan Application: Mortgage application, loan application form
- Bank Statement: Account statement, transaction history
- Tax Return: 1040, tax form, W-2, 1099
- Appraisal Report: Property appraisal, home valuation
- Title Insurance: Title policy, title commitment
- Purchase Agreement: Purchase contract, sales agreement
- Other: Doesn't fit other categories

Document sample (beginning and end):
{text_sample}

Respond with ONLY the category name, nothing else.
"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50
        )
        doc_type = response.choices[0].message.content.strip()

        valid_types = [
            'Loan Application', 'Loan Estimate', 'Closing Disclosure', 'Mortgage Note',
            'Deed of Trust', 'Title Insurance', 'Appraisal Report', 'Home Inspection',
            'Insurance Policy', 'Pay Stub', 'W-2', 'Tax Return', 'Bank Statement',
            'Asset Statement', 'Employment Verification', 'Credit Report', 'Gift Letter',
            'Purchase Agreement', 'Addendum', 'Disclosure Form', 'Driver License',
            'Social Security Card', 'Passport', 'Resume', 'Employment Contract', 'Other'
        ]

        for v_type in valid_types:
            if v_type.lower() in doc_type.lower():
                return v_type
        return 'Other'
    except Exception as e:
        print(f"Classification error: {e}")
        return 'Other'

def detect_document_boundary(prev_text: str, curr_text: str, current_doc_type: str = None) -> bool:
    if not prev_text or not curr_text:
        return False

    prev_sample = prev_text[-800:]
    curr_sample = curr_text[:800]
    prompt = f"""Determine if these two pages are from the SAME document. Current document type: '{current_doc_type or 'Unknown'}'.

End of Previous Page:
...{prev_sample}

Start of Current Page:
{curr_sample}...

Consider content continuity, formatting, and topics.
Answer ONLY 'Yes' or 'No'.
"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        return response.choices[0].message.content.strip().lower().startswith('yes')
    except Exception as e:
        print(f"Boundary detection error: {e}")
        return True

def expand_vague_query(query: str) -> Tuple[str, List[str]]:
    prompt = f"""You are a mortgage industry expert. Expand this potentially vague query into a clear, detailed question and generate related search terms.

Original Query: "{query}"

Provide:
1. An expanded, detailed version of the query
2. 3-5 related search terms or concepts

Respond in JSON format:
{{
  "expanded_query": "detailed question here",
  "search_terms": ["term1", "term2", "term3"]
}}
"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content.strip())
        expanded = result.get("expanded_query", query)
        terms = result.get("search_terms", [])
        print(f"Query expanded: '{query}' -> '{expanded}'")
        return expanded, terms
    except Exception as e:
        print(f"Query expansion error: {e}")
        return query, []

def predict_query_document_type(query: str) -> Tuple[str, float]:
    prompt = f"""Analyze this mortgage-related query and predict which document type would most likely contain the answer.

Query: "{query}"

Choose from:
Loan Application, Loan Estimate, Closing Disclosure, Mortgage Note, Deed of Trust,
Title Insurance, Appraisal Report, Home Inspection, Insurance Policy,
Pay Stub, W-2, Tax Return, Bank Statement, Asset Statement,
Employment Verification, Credit Report, Gift Letter,
Purchase Agreement, Addendum, Disclosure Form, Driver License,
Social Security Card, Passport, Employment Contract, Resume, Other

Respond in JSON: {{"type": "DocumentType", "confidence": 0.85}}
"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content.strip())
        return result.get("type", "Other"), result.get("confidence", 0.5)
    except Exception as e:
        print(f"Query routing error: {e}")
        return "Other", 0.0

def extract_and_analyze_pdf(pdf_file) -> Tuple[List[PageInfo], List[LogicalDocument]]:
    print("\nStarting PDF extraction and analysis...")

    if hasattr(pdf_file, "read"):
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    else:
        doc = fitz.open(pdf_file)

    pages_info = []
    total_ocr_pages = 0

    for i, page in enumerate(doc):
        text, ocr_used, confidence = extract_text_hybrid(page, i)
        if ocr_used:
            total_ocr_pages += 1

        pages_info.append(PageInfo(
            page_num=i,
            text=text,
            ocr_used=ocr_used,
            confidence=confidence
        ))

    doc.close()
    print(f"Extracted {len(pages_info)} pages ({total_ocr_pages} used OCR).")

    print("\nAnalyzing document structure...")
    logical_docs = []
    current_doc_pages = []
    doc_counter = 0

    for i, page_info in enumerate(pages_info):
        if i == 0:
            current_doc_type = classify_mortgage_document(page_info.text)
            page_info.doc_type = current_doc_type
            current_doc_pages = [page_info]
            print(f"  - Document 1: {current_doc_type}")
        else:
            is_same = detect_document_boundary(
                pages_info[i-1].text,
                page_info.text,
                current_doc_pages[-1].doc_type
            )
            if is_same:
                page_info.doc_type = current_doc_pages[-1].doc_type
                current_doc_pages.append(page_info)
            else:
                logical_doc = LogicalDocument(
                    doc_id=f"doc_{doc_counter}",
                    doc_type=current_doc_pages[-1].doc_type,
                    page_start=current_doc_pages[0].page_num,
                    page_end=current_doc_pages[-1].page_num,
                    text="\n\n".join([p.text for p in current_doc_pages])
                )
                logical_docs.append(logical_doc)
                doc_counter += 1

                current_doc_type = classify_mortgage_document(page_info.text)
                page_info.doc_type = current_doc_type
                current_doc_pages = [page_info]
                print(f"  - Document {doc_counter + 1}: {current_doc_type}")

    if current_doc_pages:
        logical_doc = LogicalDocument(
            doc_id=f"doc_{doc_counter}",
            doc_type=current_doc_pages[-1].doc_type,
            page_start=current_doc_pages[0].page_num,
            page_end=current_doc_pages[-1].page_num,
            text="\n\n".join([p.text for p in current_doc_pages])
        )
        logical_docs.append(logical_doc)

    print(f"Identified {len(logical_docs)} logical documents.")
    return pages_info, logical_docs

def process_all_documents(logical_docs: List[LogicalDocument]) -> List[ChunkMetadata]:
    print("\nChunking documents...")
    all_chunks = []

    splitter = SentenceSplitter(
        chunk_size=800,
        chunk_overlap=150,
        paragraph_separator="\n\n"
    )

    for logical_doc in logical_docs:
        if not logical_doc.text.strip():
            print(f"  Skipping empty document: {logical_doc.doc_type}")
            continue

        nodes = splitter.get_nodes_from_documents([Document(text=logical_doc.text)])
        chunks_for_doc = []

        for i, node in enumerate(nodes):
            chunk_text = node.get_content()

            if len(chunk_text.strip()) < 50:
                continue

            chunk_meta = ChunkMetadata(
                chunk_id=f"{logical_doc.doc_id}_chunk_{i}",
                doc_id=logical_doc.doc_id,
                doc_type=logical_doc.doc_type,
                chunk_index=i,
                page_start=logical_doc.page_start,
                page_end=logical_doc.page_end,
                text=chunk_text
            )
            chunks_for_doc.append(chunk_meta)

        logical_doc.chunks = chunks_for_doc
        all_chunks.extend(chunks_for_doc)
        print(f"  - {logical_doc.doc_type}: {len(chunks_for_doc)} chunks")

    return all_chunks

class IntelligentRetriever:
    def __init__(self):
        self.index = None
        self.chunks_metadata = []
        self.doc_type_indices = {}
        self.bm25 = None
        self.tokenized_corpus = []

    def build_indices(self, chunks_metadata: List[ChunkMetadata]):
        print("\nBuilding indices...")
        self.chunks_metadata = chunks_metadata

        if not chunks_metadata:
            print("No chunks to index!")
            return

        texts = [chunk.text for chunk in chunks_metadata]

        print("  - Encoding chunks with all-mpnet-base-v2...")
        embeddings = embed_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype('float32'))

        print("  - Building BM25 index...")
        self.tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        doc_types = set(chunk.doc_type for chunk in chunks_metadata)
        for doc_type in doc_types:
            type_indices = [i for i, chunk in enumerate(chunks_metadata) if chunk.doc_type == doc_type]
            if type_indices:
                type_embeddings = embeddings[type_indices]
                type_index = faiss.IndexFlatIP(dim)
                type_index.add(type_embeddings.astype('float32'))
                self.doc_type_indices[doc_type] = {'index': type_index, 'mapping': type_indices}

        print(f"Indexed {len(chunks_metadata)} chunks across {len(doc_types)} document types.")

    def retrieve(self, query: str, k: int = 10, filter_doc_type: Optional[str] = None,
                 auto_route: bool = True, use_reranking: bool = True) -> List[Tuple[ChunkMetadata, float]]:

        expanded_query, search_terms = expand_vague_query(query)
        combined_query = expanded_query + " " + " ".join(search_terms)

        query_embedding = embed_model.encode(
            [combined_query],
            normalize_embeddings=True
        ).astype('float32')

        initial_k = k * 3 if use_reranking else k

        if filter_doc_type and filter_doc_type in self.doc_type_indices:
            type_data = self.doc_type_indices[filter_doc_type]
            D, I = type_data['index'].search(query_embedding, min(initial_k, len(type_data['mapping'])))
            chunk_indices = [type_data['mapping'][i] for i in I[0]]
        elif auto_route:
            predicted_type, confidence = predict_query_document_type(query)
            print(f"Query routed to: {predicted_type} (confidence: {confidence:.2f})")
            if confidence > 0.7 and predicted_type in self.doc_type_indices:
                type_data = self.doc_type_indices[predicted_type]
                D, I = type_data['index'].search(query_embedding, min(initial_k, len(type_data['mapping'])))
                chunk_indices = [type_data['mapping'][i] for i in I[0]]
            else:
                D, I = self.index.search(query_embedding, initial_k)
                chunk_indices = I[0]
        else:
            D, I = self.index.search(query_embedding, initial_k)
            chunk_indices = I[0]

        semantic_scores = D[0].tolist()

        tokenized_query = combined_query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores_for_chunks = [bm25_scores[i] for i in chunk_indices]

        if semantic_scores:
            min_score = min(semantic_scores)
            max_score = max(semantic_scores)
            if max_score > min_score:
                semantic_scores = [(s - min_score) / (max_score - min_score) for s in semantic_scores]
            else:
                semantic_scores = [1.0] * len(semantic_scores)

        if bm25_scores_for_chunks:
            min_bm25 = min(bm25_scores_for_chunks)
            max_bm25 = max(bm25_scores_for_chunks)
            if max_bm25 > min_bm25:
                bm25_scores_for_chunks = [(s - min_bm25) / (max_bm25 - min_bm25) for s in bm25_scores_for_chunks]
            else:
                bm25_scores_for_chunks = [1.0] * len(bm25_scores_for_chunks)

        hybrid_scores = [0.7 * sem + 0.3 * bm25 for sem, bm25 in zip(semantic_scores, bm25_scores_for_chunks)]

        candidates = [(self.chunks_metadata[i], hybrid_scores[idx]) for idx, i in enumerate(chunk_indices)]

        if use_reranking and len(candidates) > k:
            print(f"  - Reranking top {len(candidates)} candidates...")
            pairs = [[expanded_query, chunk.text] for chunk, _ in candidates]
            rerank_scores = reranker.predict(pairs)

            reranked = sorted(
                zip(candidates, rerank_scores),
                key=lambda x: x[1],
                reverse=True
            )[:k]

            max_rerank = max([score for _, score in reranked])
            min_rerank = min([score for _, score in reranked])

            final_results = []
            for (chunk, hybrid_score), rerank_score in reranked:
                if max_rerank > min_rerank:
                    norm_rerank = (rerank_score - min_rerank) / (max_rerank - min_rerank)
                else:
                    norm_rerank = 1.0
                final_score = 0.5 * hybrid_score + 0.5 * norm_rerank
                final_results.append((chunk, final_score))

            return final_results
        else:
            return sorted(candidates, key=lambda x: x[1], reverse=True)[:k]

def generate_answer_with_sources(query: str, retrieved_chunks: List[Tuple[ChunkMetadata, float]]) -> Dict:
    if not retrieved_chunks:
        return {'answer': "I couldn't find relevant information.", 'sources': [], 'confidence': 0.0}

    context_parts = [
        f"[Source: {c.doc_type}, Pages {c.page_start+1}-{c.page_end+1}]\n{c.text}"
        for c, s in retrieved_chunks
    ]
    context = "\n\n---\n\n".join(context_parts)

    sources = [{
        'doc_type': c.doc_type,
        'pages': f"{c.page_start+1}-{c.page_end+1}",
        'relevance': f"{s:.1%}",
        'preview': c.text[:150] + "..."
    } for c, s in retrieved_chunks]

    prompt = f"""You are an expert mortgage document analyst. Use the provided context to answer questions accurately.

RULES:
- Base answers ONLY on the provided context
- Cite specific document types and pages
- Extract exact numbers, dates, names, and monetary amounts
- If information is missing, state that clearly
- For financial data, preserve exact formatting (dollar amounts, percentages)
- If contradictory information exists, mention it

Context:
{context}

Question: {query}

Answer:
"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000
        )
        answer = response.choices[0].message.content.strip()
        avg_score = sum(s for _, s in retrieved_chunks) / len(retrieved_chunks)

        return {
            'answer': answer,
            'sources': sources,
            'confidence': avg_score,
            'chunks_used': len(retrieved_chunks)
        }
    except Exception as e:
        print(f"Answer generation error: {e}")
        return {
            'answer': f"Error generating answer: {e}",
            'sources': sources,
            'confidence': 0.0
        }

class EnhancedDocumentStore:
    def __init__(self):
        self.is_ready = False
        self.retriever = IntelligentRetriever()
        self.pages_info = []
        self.logical_docs = []
        self.chunks_metadata = []
        self.processing_stats = {}

    def process_pdf(self, pdf_file, filename: str = "document.pdf"):
        self.is_ready = False
        start_time = datetime.now()
        try:
            self.pages_info, self.logical_docs = extract_and_analyze_pdf(pdf_file)
            self.chunks_metadata = process_all_documents(self.logical_docs)
            self.retriever.build_indices(self.chunks_metadata)

            process_time = (datetime.now() - start_time).total_seconds()

            ocr_pages = sum(1 for p in self.pages_info if p.ocr_used)
            avg_confidence = np.mean([p.confidence for p in self.pages_info])

            self.processing_stats = {
                'filename': filename,
                'total_pages': len(self.pages_info),
                'documents_found': len(self.logical_docs),
                'total_chunks': len(self.chunks_metadata),
                'document_types': list(set(doc.doc_type for doc in self.logical_docs)),
                'processing_time': f"{process_time:.1f}s",
                'ocr_pages': ocr_pages,
                'avg_confidence': f"{avg_confidence:.1%}"
            }
            self.is_ready = True
            return True, self.processing_stats
        except Exception as e:
            print(f"Error in process_pdf: {e}")
            import traceback
            traceback.print_exc()
            return False, {'error': str(e)}

    def query(self, question: str, filter_type: Optional[str] = None, auto_route: bool = True, k: int = 5) -> Dict:
        if not self.is_ready:
            return {
                'answer': "Please upload and process a PDF first.",
                'sources': [],
                'confidence': 0.0
            }

        retrieved = self.retriever.retrieve(
            question,
            k=k,
            filter_doc_type=filter_type,
            auto_route=auto_route,
            use_reranking=True
        )
        result = generate_answer_with_sources(question, retrieved)
        result['filter_used'] = filter_type or ('auto' if auto_route else 'none')
        return result

    def get_document_structure(self) -> List[Dict]:
        return [{
            'type': doc.doc_type,
            'pages': f"{doc.page_start + 1}-{doc.page_end + 1}",
            'chunks': len(doc.chunks) if doc.chunks else 0
        } for doc in self.logical_docs]

def get_document_store():
    if not hasattr(get_document_store, 'instance'):
        get_document_store.instance = EnhancedDocumentStore()
    return get_document_store.instance