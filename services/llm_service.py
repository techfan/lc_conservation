from typing import AsyncGenerator, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatZhipuAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.tools import Tool
from config import settings
from services.rag_service import rag_service
from services.sql_service import sql_service
from context.manager import ContextManager
from db.redis_client import redis_client
from schemas.schemas import MessageRole


class LLMService:
    def __init__(self):
        self._llm = None
        self._context_manager = ContextManager()
        self._tools = None
    
    def initialize(self):
        self._llm = ChatZhipuAI(
            model=settings.models.llm.model_name,
            api_key=settings.models.llm.api_key,
            temperature=0.1
        )
        
        self._tools = [
            Tool(
                name="rag_search",
                func=self._rag_search,
                description="检索知识库获取相关知识",
            ),
            Tool(
                name="sql_query",
                func=self._sql_query,
                description="执行SQL查询获取数据",
            )
        ]
    
    def _rag_search(self, query: str) -> str:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(rag_service.search(query))
            return "\n".join([f"{r['content']}" for r in results])
        finally:
            loop.close()
    
    def _sql_query(self, sql: str) -> str:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(sql_service.execute_query(sql))
            return str(results)
        finally:
            loop.close()
    
    async def stream_chat(
        self,
        question: str,
        session_id: str,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        messages = await redis_client.get_messages(session_id)
        db_schema = await sql_service.get_db_schema()
        
        context_assembly = await self._context_manager.assemble_context(
            db_schema=db_schema,
            user_info={"user_id": user_id},
            conversation_history=messages
        )
        
        system_template = context_assembly.system_prompt + (
            "\n\n可用工具:\n{tools}\n\n"
            "你必须严格按照以下格式输出:\n\n"
            "Question: 用户的问题\n"
            "Thought: 思考下一步该做什么\n"
            "Action: 工具名称，必须是 [{tool_names}] 之一\n"
            "Action Input: 工具的输入参数\n"
            "Observation: 工具返回的结果\n"
            "... (以上 Thought/Action/Action Input/Observation 可重复多次)\n"
            "Thought: 我现在知道最终答案\n"
            "Final Answer: 对用户问题的最终回答\n\n"
            "注意: Action 后面只能跟 [{tool_names}] 中的工具名，不能编造工具名。\n"
            "如果不使用工具，直接输出 Final Answer。"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("ai", "{agent_scratchpad}"),
        ])

        agent = create_react_agent(self._llm, self._tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, tools=self._tools, verbose=True, handle_parsing_errors=True)
        
        chat_history = []
        for msg in messages:
            if msg["role"] == "user":
                chat_history.append(("user", msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(("assistant", msg["content"]))
        
        async for step in agent_executor.astream(
            {"input": question, "chat_history": chat_history},
        ):
            if "output" in step:
                yield step["output"]
        
        await redis_client.add_message(session_id, {
            "role": MessageRole.USER,
            "content": question
        })
    
    async def chat(
        self,
        question: str,
        session_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        full_answer = ""
        sources = []
        
        async for chunk in self.stream_chat(question, session_id, user_id):
            full_answer += chunk
        
        await redis_client.add_message(session_id, {
            "role": MessageRole.ASSISTANT,
            "content": full_answer
        })
        
        return {
            "answer": full_answer,
            "sources": sources,
            "thinking_process": None
        }


llm_service = LLMService()
