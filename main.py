import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from config import settings
from exception.exceptions import ServiceException
from routers import chat, document, schema, session
from schemas.schemas import ResponseObject
from services.llm_service import llm_service
from services.rag_service import rag_service
from db.redis_client import redis_client
from db.database import engine
from models.models import Base

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("应用启动中...")
    
    try:
        await redis_client.connect()
        logger.info("Redis连接成功")
    except Exception as e:
        logger.error(f"Redis连接失败: {e}")
    
    try:
        rag_service.connect()
        logger.info("RAG服务连接成功")
    except Exception as e:
        logger.error(f"RAG服务连接失败: {e}")
    
    try:
        llm_service.initialize()
        logger.info("LLM服务初始化成功")
    except Exception as e:
        logger.error(f"LLM服务初始化失败: {e}")
    
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("数据库初始化成功")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
    
    logger.info(f"{settings.project.name}【{settings.project.active}】启动完成")
    yield
    
    logger.info("应用关闭中...")
    try:
        await redis_client.disconnect()
        logger.info("Redis断开成功")
    except Exception as e:
        logger.error(f"Redis断开失败: {e}")
    logger.info("应用关闭完成")


app = FastAPI(
    title=settings.project.name,
    version=settings.project.version,
    lifespan=lifespan
)

app.include_router(chat.api)
app.include_router(document.api)
app.include_router(schema.api)
app.include_router(session.api)


### 全局异常处理器 ###
@app.exception_handler(ServiceException)
async def service_exception_handler(request: Request, exc: ServiceException):
    """处理自定义服务异常"""
    # logger.error(f"ServiceException: {str(exc)}")
    logger.exception("业务异常！")
    
    error_response = ResponseObject(
        success=False,
        message=exc.message
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """处理所有未捕获的异常"""
    logger.exception(f"未知异常！")

    error_response = ResponseObject(
        success=False,
        message=str(exc)
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump()
    )


@app.get("/")
async def root():
    return {
        "service": settings.project.name,
        "version": settings.project.version
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
