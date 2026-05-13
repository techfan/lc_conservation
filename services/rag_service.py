import logging
import os
import tempfile
import uuid

from typing import List, Dict, Any, Optional

from fastapi import UploadFile, File
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from pymilvus import MilvusClient, DataType
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
from config import settings
from exception.exceptions import ServiceException

logger = logging.getLogger(__name__)

class OllamaEmbeddings:
    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def embed_query(self, text: str) -> List[float]:
        try:
            response = ollama.embeddings(model=self.model, prompt=text)
            return response["embedding"]
        except Exception as e:
            logger.error(f"使用 ollama 生成嵌入向量失败: {e}")
            import random
            return [random.uniform(-1, 1) for _ in range(1024)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings


class OllamaReranker:
    def __init__(self, model: str = "bge-reranker-v2-m3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not documents:
            return []

        try:
            scored_docs = []
            for doc in documents:
                content = doc.get("content", "")
                prompt = f"Query: {query}\nDocument: {content}\nRelevance score (0-1):"
                response = ollama.generate(model=self.model, prompt=prompt)
                try:
                    score = float(response["response"].strip())
                except:
                    score = doc.get("score", 0)

                scored_docs.append({
                    **doc,
                    "rerank_score": score
                })

            scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored_docs[:top_k]

        except Exception as e:
            logger.error(f"使用 ollama 重排序失败: {e}")
            return documents[:top_k]


class RAGService:
    def __init__(self):
        self._client: Optional[MilvusClient] = None
        self._collection_name: str = settings.milvus.collection
        self._embeddings: Optional[OllamaEmbeddings] = None
        self._reranker: Optional[OllamaReranker] = None
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )

    def connect(self):
        self._client = MilvusClient(uri=f"http://{settings.milvus.host}:{settings.milvus.port}")
        self._create_collection()
        self._embeddings = OllamaEmbeddings(model="bge-m3")
        self._reranker = OllamaReranker(model="qllama/bge-reranker-v2-m3")

    def _create_collection(self):
        target_collection = self._collection_name

        try:
            all_collections = self._client.list_collections()
            logger.info(f"Milvus 中所有集合: {all_collections}")
        except Exception as e:
            logger.error(f"获取集合列表失败: {e}")
            all_collections = []

        logger.info(f"使用集合: {target_collection}")

        if not self._client.has_collection(target_collection):
            logger.info(f"集合 {target_collection} 不存在，正在创建...")
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=256)
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="metadata", datatype=DataType.JSON)

            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )

            self._client.create_collection(
                collection_name=target_collection,
                schema=schema,
                index_params=index_params
            )
            logger.info(f"集合 {target_collection} 创建成功")

        self._client.load_collection(target_collection)
        logger.info(f"集合 {target_collection} 已成功加载")

    async def add_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None):
        chunks = self._text_splitter.split_text(content)
        data = []

        for i, chunk in enumerate(chunks):
            embedding = self._embeddings.embed_query(chunk)
            chunk_id = f"{doc_id}_{i}"
            data.append({
                "id": chunk_id,
                "content": chunk,
                "metadata": metadata or {},
                "embedding": embedding
            })

        self._client.insert(collection_name=self._collection_name, data=data)
        self._client.flush(collection_name=self._collection_name)

    async def search(self, query: str, top_k: int = 5, retrieve_k: int = 20) -> List[Dict[str, Any]]:
        query_embedding = self._embeddings.embed_query(query)
        results = self._client.search(
            collection_name=self._collection_name,
            data=[query_embedding],
            anns_field="embedding",
            search_params={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=retrieve_k,
            output_fields=["content", "metadata"]
        )

        docs = []
        for hits in results:
            for hit in hits:
                docs.append({
                    "content": hit.get("entity", {}).get("content", ""),
                    "metadata": hit.get("entity", {}).get("metadata", {}),
                    "score": hit.get("distance", 0)
                })

        if self._reranker and docs:
            docs = self._reranker.rerank(query, docs, top_k)

        return docs[:top_k]

    async def delete_document(self, doc_id: str):
        expr = f"id like '{doc_id}_%'"
        self._client.delete(collection_name=self._collection_name, filter=expr)

    async def list_documents(self) -> dict[str, list[str]]:
        results = self._client.query(
            collection_name=self._collection_name,
            filter="id != ''",
            output_fields=["id"]
        )
        doc_ids = set()
        for res in results:
            base_id = res["id"].rsplit("_", 1)[0]
            doc_ids.add(base_id)
        return {"documents": list(doc_ids)}


rag_service = RAGService()


async def upload_document(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            full_content = "\n".join([doc.page_content for doc in docs])
        elif file.filename.endswith('.docx'):
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()
            full_content = "\n".join([doc.page_content for doc in docs])
        else:
            # 对于文本文件，尝试多种编码读取
            full_content = ""
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            for encoding in encodings:
                try:
                    with open(tmp_path, 'r', encoding=encoding) as f:
                        full_content = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if not full_content:
                raise ServiceException(message="无法解析文件编码")

        await rag_service.add_document(
            doc_id=doc_id,
            content=full_content,
            metadata={"file_name": file.filename}
        )

        return {"document_id": doc_id, "file_name": file.filename}
    finally:
        os.unlink(tmp_path)