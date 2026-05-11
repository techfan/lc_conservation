import logging
from typing import Dict, Any, List, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import AsyncSessionLocal

logger = logging.getLogger(__name__)


class SQLService:
    def __init__(self):
        self._schema_cache: Optional[str] = ""
    
    async def get_db_schema(self) -> str:
        if self._schema_cache:
            return self._schema_cache
        
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(text("""
                    SELECT table_name, column_name, data_type, column_default, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    ORDER BY table_name, ordinal_position
                """))
                rows = result.fetchall()
            
            schema_dict: Dict[str, List[Dict]] = {}
            for row in rows:
                table = row[0]
                if table not in schema_dict:
                    schema_dict[table] = []
                schema_dict[table].append({
                    "column": row[1],
                    "type": row[2],
                    "default": row[3],
                    "nullable": row[4]
                })
            
            schema_lines = []
            for table, columns in schema_dict.items():
                schema_lines.append(f"表名: {table}")
                for col in columns:
                    schema_lines.append(f"  - {col['column']}: {col['type']}")
                schema_lines.append("")
            
            self._schema_cache = "\n".join(schema_lines)
            return self._schema_cache
        except Exception as e:
            logger.error(f"获取数据库schema失败: {e}")
            # 返回空的schema，让系统继续运行
            self._schema_cache = "数据库schema（暂时不可用）"
            return self._schema_cache
    
    async def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
        sql_upper = sql.upper()
        for keyword in forbidden_keywords:
            if keyword in sql_upper:
                raise ValueError(f"禁止执行的SQL操作: {keyword}")
        
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(text(sql))
                rows = result.fetchall()
                columns = result.keys()
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"执行SQL查询失败: {e}")
            return [{"error": "SQL查询失败", "message": str(e)}]
    
    def clear_schema_cache(self):
        self._schema_cache = ""


sql_service = SQLService()
