from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_name: Optional[str] = None
    tool_result: Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    session_id: str
    user_id: Optional[str] = None
    stream: bool = True


class SourceReference(BaseModel):
    type: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceReference]
    thinking_process: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None


class DocumentUploadRequest(BaseModel):
    file_name: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentInfo(BaseModel):
    id: str
    file_name: str
    metadata: Dict[str, Any]
    created_at: datetime


class SessionInfo(BaseModel):
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int
