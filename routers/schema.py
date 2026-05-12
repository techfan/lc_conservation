import logging

from fastapi import APIRouter

from schemas.schemas import ResponseObject
from services.sql_service import sql_service

api = APIRouter(prefix="/schema")
log = logging.getLogger(__name__)

@api.get("/refresh")
async def refresh_schema():
    log.info("开始刷新schema")
    sql_service.clear_schema_cache()
    schema = await sql_service.get_db_schema()
    return ResponseObject(data={"schema": schema})