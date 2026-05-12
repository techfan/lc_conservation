import logging

from fastapi import APIRouter

from schemas.schemas import ResponseObject
from services import session_service

api = APIRouter(prefix="/session")
log = logging.getLogger(__name__)

@api.get("/get/{session_id}")
async def get_session(session_id: str):
    log.info(f"开始获取会话: {session_id}")
    data = await session_service.get_session(session_id)
    return ResponseObject(data=data)


@api.delete("/delete/{session_id}")
async def clear_session(session_id: str):
    log.info(f"开始删除会话: {session_id}")
    await session_service.clear_session(session_id)
    return ResponseObject(data={"session_id": session_id})