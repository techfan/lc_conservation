import logging

from fastapi import APIRouter, UploadFile, File

from schemas.schemas import ResponseObject
from services import rag_service
from services.rag_service import rag_service

api = APIRouter(prefix="/documents")
log = logging.getLogger(__name__)


@api.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    log.info(f"开始上传文件: {file.filename}")
    data = await rag_service.upload_document(file)
    return ResponseObject(data=data)


@api.get("/list")
async def list_documents():
    log.info("开始列出所有文件")
    data = await rag_service.list_documents()
    return ResponseObject(data=data)


@api.delete("/delete/{doc_id}")
async def delete_document(doc_id: str):
    log.info(f"开始删除文件: {doc_id}")
    await rag_service.delete_document(doc_id)
    return ResponseObject(message="删除成功")