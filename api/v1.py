from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import List, Optional
import uuid
from datetime import datetime
from schemas.schemas import (
    QueryRequest, QueryResponse, DocumentInfo,
    SessionInfo, SourceReference
)
from services.llm_service import llm_service
from services.rag_service import rag_service
from db.sql_service import sql_service
from db.redis_client import redis_client
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import tempfile
import os

router = APIRouter()


@router.post("/query")
async def query(request: QueryRequest):
    if request.stream:
        async def generate():
            async for chunk in llm_service.stream_chat(
                request.question,
                request.session_id,
                request.user_id
            ):
                yield chunk
        return StreamingResponse(generate(), media_type="text/plain")
    else:
        result = await llm_service.chat(
            request.question, request.session_id, request.user_id)
        return QueryResponse(
            answer=result["answer"],
            sources=[SourceReference(**s) for s in result["sources"]],
            thinking_process=result.get("thinking_process")
        )


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            full_content = "\n".join([doc.page_content for doc in docs])
        elif file.filename.endswith('.docx'):
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()
            full_content = "\n".join([doc.page_content for doc in docs])
        else:
            # 对于文本文件，尝试多种编码读取
            full_content = ""
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            for encoding in encodings:
                try:
                    with open(tmp_path, 'r', encoding=encoding) as f:
                        full_content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if not full_content:
                raise HTTPException(status_code=400, detail="无法解析文件编码")
        
        await rag_service.add_document(
            doc_id=doc_id,
            content=full_content,
            metadata={"file_name": file.filename}
        )
        
        return {"document_id": doc_id, "file_name": file.filename}
    finally:
        os.unlink(tmp_path)


@router.get("/documents")
async def list_documents():
    doc_ids = await rag_service.list_documents()
    return {"documents": doc_ids}


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    await rag_service.delete_document(doc_id)
    return {"message": "Document deleted"}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = await redis_client.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionInfo(
        session_id=session_id,
        user_id=session.get("user_id"),
        created_at=session.get("created_at"),
        updated_at=session.get("updated_at"),
        message_count=len(session.get("messages", []))
    )


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    await redis_client.clear_session(session_id)
    return {"message": "Session cleared"}


@router.get("/schema/refresh")
async def refresh_schema():
    sql_service.clear_schema_cache()
    schema = await sql_service.get_db_schema()
    return {"schema": schema}
