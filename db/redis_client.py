import redis.asyncio as redis
from typing import Optional, Dict, Any
import json
from datetime import timedelta
from config import settings


class RedisClient:
    def __init__(self):
        self._client: Optional[redis.Redis] = None
    
    async def connect(self):
        if not self._client:
            self._client = redis.from_url(
                settings.redis.url,
                decode_responses=True
            )
    
    async def disconnect(self):
        if self._client:
            await self._client.close()
            self._client = None
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        data = await self._client.get(f"session:{session_id}")
        return json.loads(data) if data else None
    
    async def save_session(self, session_id: str, session_data: Dict[str, Any], ttl: Optional[int] = None):
        ttl = ttl or settings.redis.ttl
        await self._client.setex(
            f"session:{session_id}",
            timedelta(seconds=ttl),
            json.dumps(session_data, ensure_ascii=False, default=str)
        )
    
    async def add_message(self, session_id: str, message: Dict[str, Any]):
        session = await self.get_session(session_id) or {"messages": [], "created_at": None}
        session["messages"].append(message)
        session["updated_at"] = None
        await self.save_session(session_id, session)
    
    async def get_messages(self, session_id: str) -> list:
        session = await self.get_session(session_id)
        return session.get("messages", []) if session else []
    
    async def clear_session(self, session_id: str):
        await self._client.delete(f"session:{session_id}")
    
    async def cache_context(self, key: str, data: Any, ttl: int = 3600):
        await self._client.setex(
            f"context:{key}",
            timedelta(seconds=ttl),
            json.dumps(data, ensure_ascii=False, default=str)
        )
    
    async def get_cached_context(self, key: str) -> Optional[Any]:
        data = await self._client.get(f"context:{key}")
        return json.loads(data) if data else None


redis_client = RedisClient()
