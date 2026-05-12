import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from schemas.schemas import QueryRequest, SourceReference, ResponseObject
from services.llm_service import llm_service

api = APIRouter()
log = logging.getLogger(__name__)

@api.post("/query")
async def query(request: QueryRequest):
    log.info(f"问答请求入参：{request.model_dump_json()}")
    if request.stream:
        async def generate():
            async for chunk in llm_service.stream_chat(request.question,request.session_id,request.user_id):
                yield chunk
        return StreamingResponse(generate(), media_type="text/plain")

    else:
        result = await llm_service.chat(
            request.question, request.session_id, request.user_id)

        data = {"answer": result["answer"],
                "sources": [SourceReference(**s) for s in result["sources"]],
                "thinking_process": result.get("thinking_process")}

        return ResponseObject(data=data)