from fastapi import FastAPI
from contextlib import asynccontextmanager
from config.settings import settings
from api.v1 import router as api_router
from services.llm_service import llm_service
from services.rag_service import rag_service
from db.redis_client import redis_client
from db.database import engine
from models.models import Base
from config.logger import logger


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
    
    logger.info("应用启动完成")
    yield
    
    logger.info("应用关闭中...")
    try:
        await redis_client.disconnect()
        logger.info("Redis断开成功")
    except Exception as e:
        logger.error(f"Redis断开失败: {e}")
    logger.info("应用关闭完成")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
