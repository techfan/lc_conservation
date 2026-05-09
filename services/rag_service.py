from typing import List, Dict, Any, Optional
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama
from config.settings import settings
from config.logger import logger


class OllamaEmbeddings:
    """使用 ollama 本地安装的 bge-m3 模型"""
    
    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def embed_query(self, text: str) -> List[float]:
        """为查询文本生成嵌入向量"""
        try:
            response = ollama.embeddings(model=self.model, prompt=text)
            return response["embedding"]
        except Exception as e:
            logger.error(f"使用 ollama 生成嵌入向量失败: {e}")
            # 如果 ollama 失败，返回随机向量作为备用
            import random
            return [random.uniform(-1, 1) for _ in range(1024)]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档列表生成嵌入向量"""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings


class OllamaReranker:
    """使用 ollama 本地安装的 bge-reranker-v2-m3 模型进行重排序"""
    
    def __init__(self, model: str = "bge-reranker-v2-m3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """对文档进行重排序"""
        if not documents:
            return []
        
        try:
            # 构建重排序的 prompt
            scored_docs = []
            for doc in documents:
                content = doc.get("content", "")
                # 使用 ollama 生成相关性评分
                # 注意：这只是一个简单的实现，实际使用时需要根据模型特性调整
                prompt = f"Query: {query}\nDocument: {content}\nRelevance score (0-1):"
                response = ollama.generate(model=self.model, prompt=prompt)
                # 尝试解析评分
                try:
                    score = float(response["response"].strip())
                except:
                    score = doc.get("score", 0)  # 使用原有的分数
                
                scored_docs.append({
                    **doc,
                    "rerank_score": score
                })
            
            # 按重排序分数降序排列
            scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored_docs[:top_k]
        
        except Exception as e:
            logger.error(f"使用 ollama 重排序失败: {e}")
            # 如果重排序失败，返回原始文档
            return documents[:top_k]


class RAGService:
    def __init__(self):
        self._collection: Optional[Collection] = None
        self._embeddings: Optional[OllamaEmbeddings] = None
        self._reranker: Optional[OllamaReranker] = None
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
    
    def connect(self):
        connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        self._create_collection()
        # 使用 ollama 本地安装的 bge-m3 嵌入模型
        self._embeddings = OllamaEmbeddings(model="bge-m3")
        # 使用 ollama 本地安装的 bge-reranker-v2-m3 重排序模型
        self._reranker = OllamaReranker(model="bge-reranker-v2-m3")
    
    def _create_collection(self):
        from pymilvus import utility
        
        # 首先查看 Milvus 中已有的集合
        try:
            all_collections = utility.list_collections()
            logger.info(f"Milvus 中所有集合: {all_collections}")
        except Exception as e:
            logger.error(f"获取集合列表失败: {e}")
            all_collections = []
        
        # 直接使用已有的 ai_query_docs 集合
        target_collection = "ai_query_docs"
        
        logger.info(f"使用集合: {target_collection}")
        try:
            self._collection = Collection(target_collection)
            logger.info(f"成功加载集合: {target_collection}")
        except Exception as e:
            logger.error(f"加载集合 {target_collection} 失败: {e}")
            raise
        
        # 确保索引存在
        try:
            if len(self._collection.indexes) == 0:
                logger.info(f"为集合 {target_collection} 创建向量索引")
                index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "IP",
                    "params": {"nlist": 128}
                }
                self._collection.create_index(field_name="embedding", index_params=index_params)
                logger.info("索引创建成功")
        except Exception as e:
            logger.error(f"索引创建或检查失败: {e}")
        
        # 加载集合到内存
        try:
            self._collection.load()
            logger.info(f"集合 {target_collection} 已成功加载")
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
    
    async def add_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None):
        chunks = self._text_splitter.split_text(content)
        
        for i, chunk in enumerate(chunks):
            embedding = self._embeddings.embed_query(chunk)
            chunk_id = f"{doc_id}_{i}"
            self._collection.insert([
                [chunk_id],
                [chunk],
                [metadata or {}],
                [embedding]
            ])
        self._collection.flush()
    
    async def search(self, query: str, top_k: int = 5, retrieve_k: int = 20) -> List[Dict[str, Any]]:
        """
        搜索相关文档，使用 ollama 的 bge-m3 进行检索，使用 bge-reranker-v2-m3 进行重排序
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            retrieve_k: 检索阶段返回的文档数量（用于重排序）
        
        Returns:
            重排序后的文档列表
        """
        query_embedding = self._embeddings.embed_query(query)
        results = self._collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=retrieve_k,
            output_fields=["content", "metadata"]
        )
        
        docs = []
        for hits in results:
            for hit in hits:
                docs.append({
                    "content": hit.entity.get("content"),
                    "metadata": hit.entity.get("metadata"),
                    "score": hit.score
                })
        
        # 使用 ollama 的 bge-reranker-v2-m3 进行重排序
        if self._reranker and docs:
            docs = self._reranker.rerank(query, docs, top_k)
        
        return docs[:top_k]
    
    async def delete_document(self, doc_id: str):
        expr = f"id like '{doc_id}_%'"
        self._collection.delete(expr)
    
    async def list_documents(self) -> List[str]:
        results = self._collection.query(expr="id != ''", output_fields=["id"])
        doc_ids = set()
        for res in results:
            base_id = res["id"].rsplit("_", 1)[0]
            doc_ids.add(base_id)
        return list(doc_ids)


rag_service = RAGService()
