from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import tiktoken


@dataclass
class ContextLayer:
    name: str
    content: str
    priority: int
    token_count: int = 0


class BaseContextProvider(ABC):
    @abstractmethod
    async def get_context(self, **kwargs) -> ContextLayer:
        pass
    
    def estimate_tokens(self, text: str) -> int:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except:
            return len(text) // 3


class SystemPromptLayer(BaseContextProvider):
    async def get_context(self, **kwargs) -> ContextLayer:
        content = """你是一个智能问答助手，具有以下能力：
1. RAG知识库检索 - 当需要专业知识时使用
2. SQL数据库查询 - 当需要数据查询时使用

规则：
- 严格按照ReAct模式执行：思考→行动→观察→验证
- 禁止编造信息，只能使用检索到的内容回答
- SQL生成时防止SQL注入
- 回答要附带来源引用

工具清单：
- rag_search: 检索知识库
- sql_query: 查询数据库
"""
        return ContextLayer(
            name="system_prompt",
            content=content,
            priority=1,
            token_count=self.estimate_tokens(content)
        )


class SystemContextLayer(BaseContextProvider):
    def __init__(self, db_schema: str = ""):
        self.db_schema = db_schema
    
    async def get_context(self, **kwargs) -> ContextLayer:
        content = f"""数据库Schema：
{self.db_schema}

业务规则：
- 查询结果限制最多100条
- 禁止执行修改操作
- 敏感字段需脱敏
"""
        return ContextLayer(
            name="system_context",
            content=content,
            priority=2,
            token_count=self.estimate_tokens(content)
        )


class UserContextLayer(BaseContextProvider):
    def __init__(self, user_info: Optional[Dict[str, Any]] = None):
        self.user_info = user_info or {}
    
    async def get_context(self, **kwargs) -> ContextLayer:
        content = f"""用户信息：
用户ID: {self.user_info.get('user_id', 'anonymous')}
角色: {self.user_info.get('role', 'user')}
偏好: {self.user_info.get('preferences', {})}
"""
        return ContextLayer(
            name="user_context",
            content=content,
            priority=3,
            token_count=self.estimate_tokens(content)
        )


class ConversationHistoryLayer(BaseContextProvider):
    def __init__(self, history: list = None):
        self.history = history or []
    
    async def get_context(self, **kwargs) -> ContextLayer:
        lines = []
        for msg in self.history:
            lines.append(f"{msg['role']}: {msg['content']}")
        content = "\n".join(lines)
        return ContextLayer(
            name="conversation_history",
            content=content,
            priority=4,
            token_count=self.estimate_tokens(content)
        )
