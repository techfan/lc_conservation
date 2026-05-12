# 企业级智能问答系统

基于 Python、FastAPI、LangChain 构建的智能问答系统，支持 RAG 知识库检索和数据库 SQL 查询。

## 技术栈

- 后端框架: FastAPI 0.100+
- AI框架: LangChain
- 数据库: PostgreSQL 14+
- 向量存储: Milvus
- 缓存/会话: Redis
- Embedding模型: bge-m3
- LLM: glm-4.7-flash

## 项目结构

```
lc_conservation/
├── api/
│   ├── __init__.py
│   └── v1.py               # API路由
├── config/
│   ├── __init__.py
│   └── settings.py         # 配置
├── context/
│   ├── __init__.py
│   ├── layers.py           # 上下文层定义
│   └── manager.py          # 上下文管理器
├── db/
│   ├── __init__.py
│   ├── database.py         # PostgreSQL连接
│   ├── redis_client.py     # Redis客户端
│   └── sql_service.py      # SQL查询服务
├── models/
│   ├── __init__.py
│   └── models.py           # 数据模型
├── schemas/
│   ├── __init__.py
│   └── schemas.py          # Pydantic schemas
├── services/
│   ├── __init__.py
│   ├── llm_service.py      # LLM + ReAct Agent
│   └── rag_service.py      # RAG服务
├── main.py                 # 主入口
├── requirements.txt
└── README.md
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境配置

确保以下服务已启动：
- Redis: localhost:6379 (密码: 123456)
- Milvus: localhost:19530
- PostgreSQL: localhost:5432 (数据库: ai_query)

## 启动服务

```bash
python main.py
```

或使用uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API接口

### 1. 智能问答

```bash
POST /routers/v1/query
Content-Type: application/json

{
  "question": "你的问题",
  "session_id": "会话ID",
  "user_id": "用户ID",
  "stream": true
}
```

### 2. 上传文档

```bash
POST /routers/v1/documents/upload
Content-Type: multipart/form-data

file: [PDF/Word/TXT文件]
```

### 3. 列出文档

```bash
GET /routers/v1/documents
```

### 4. 删除文档

```bash
DELETE /routers/v1/documents/{doc_id}
```

### 5. 获取会话信息

```bash
GET /routers/v1/sessions/{session_id}
```

### 6. 清除会话

```bash
DELETE /routers/v1/sessions/{session_id}
```

### 7. 刷新数据库Schema

```bash
GET /routers/v1/schema/refresh
```

## 核心功能

- ReAct模式智能工具路由
- RAG知识库检索
- 数据库SQL查询
- 分层上下文管理
- 会话管理与缓存
- 流式响应支持
