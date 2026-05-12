from db.redis_client import redis_client
from exception.exceptions import ServiceException
from schemas.schemas import SessionInfo


async def get_session(session_id: str):
    session = await redis_client.get_session(session_id)
    if not session:
        raise ServiceException(message="没有查到会话信息")

    return SessionInfo(
            session_id=session_id,
            user_id=session.get("user_id"),
            created_at=session.get("created_at"),
            updated_at=session.get("updated_at"),
            message_count=len(session.get("messages", []))
    )


async def clear_session(session_id: str):
    await redis_client.clear_session(session_id)