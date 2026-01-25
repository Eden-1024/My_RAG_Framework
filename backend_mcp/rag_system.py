# simple_rag.py
import asyncio
import logging
from typing import List, Dict, Any
from config_fixed_final import Config
from milvus_manager_fixed_final import MilvusLiteManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleRAGSystem:
    """简化的 RAG 系统 - 最终版"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.vector_store = MilvusLiteManager(self.config)
        self.initialized = False
    
    def initialize(self):
        """初始化系统"""
        if self.initialized:
            return True
            
        logger.info("正在初始化 RAG 系统...")
        
        # 连接 Milvus-Lite
        if not self.vector_store.connect():
            logger.error("无法连接到 Milvus-Lite")
            return False
        
        # 创建集合
        try:
            if not self.vector_store.create_collection():
                logger.warning("集合创建失败或已存在")
        except Exception as e:
            logger.warning(f"创建集合时出错: {e}")
        
        self.initialized = True
        logger.info("RAG 系统初始化成功！")
        return True
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """添加文档"""
        if not self.initialized:
            self.initialize()
        
        try:
            success = self.vector_store.add_documents(documents)
            if success:
                logger.info(f"成功添加 {len(documents)} 个文档")
            else:
                logger.error(f"添加文档失败")
            return success
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索文档"""
        if not self.initialized:
            self.initialize()
        
        return self.vector_store.search(query, top_k)
    
    def query_with_context(self, query: str) -> str:
        """带上下文的查询"""
        results = self.search(query, top_k=3)
        
        if not results:
            return "抱歉，我没有找到相关的信息。"
        
        # 构建上下文
        context = "以下是从知识库中找到的相关信息：\n\n"
        for i, result in enumerate(results, 1):
            score = result.get('score', 0)
            # 转换距离为相似度 (L2距离越小越相似)
            similarity = max(0, 1 - score)
            context += f"{i}. 【来源：{result.get('source', 'unknown')}，相似度：{similarity:.2%}】\n"
            context += f"   {result.get('text', '')}\n\n"
        
        # 构建回答
        answer = f"问题：{query}\n\n"
        answer += "回答：\n"
        answer += results[0].get('text', '') + "\n\n"
        
        if len(results) > 1:
            answer += "其他相关信息：\n"
            for i, result in enumerate(results[1:], 2):
                text = result.get('text', '')
                if len(text) > 100:
                    text = text[:100] + "..."
                answer += f"{i}. {text}\n"
        
        return answer
    
    def get_stats(self):
        """获取系统统计信息"""
        if not self.initialized:
            self.initialize()
        
        return self.vector_store.get_collection_info()