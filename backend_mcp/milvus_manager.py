# milvus_manager.py
import os
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import logging

logger = logging.getLogger(__name__)

class MilvusLiteManager:
    """Milvus-Lite 管理器 - 最终修复版"""
    
    def __init__(self, config):
        self.config = config
        self.embedding_model = SentenceTransformer(config.embedding_model)
        self.client = None
        self.collection_name = config.collection_name
        self.dim = config.embedding_dim
        
        # Milvus-Lite 特定配置
        self.db_path = config.milvus_db_path
    
    def connect(self):
        """连接到 Milvus-Lite"""
        try:
            # 对于 Milvus-Lite，我们使用本地文件存储
            self.client = MilvusClient(uri=self.db_path)
            logger.info(f"连接到 Milvus-Lite 数据库: {self.db_path}")
            
            # 测试连接
            collections = self.client.list_collections()
            logger.info(f"连接测试成功，现有集合: {collections}")
            return True
            
        except Exception as e:
            logger.error(f"连接 Milvus-Lite 失败: {e}")
            return False
    
    def create_collection(self):
        """创建集合 - 修复版"""
        try:
            # 检查集合是否已存在
            collections = self.client.list_collections()
            if self.collection_name in collections:
                logger.info(f"集合 '{self.collection_name}' 已存在")
                
                # 获取集合信息以验证维度
                try:
                    info = self.client.describe_collection(self.collection_name)
                    logger.info(f"集合信息: {info}")
                    
                    # 检查维度是否匹配
                    if hasattr(info, 'schema') and info.schema:
                        for field in info.schema.fields:
                            if field.name == "vector" and hasattr(field, 'params'):
                                dim = field.params.get('dim')
                                if dim and dim != self.dim:
                                    logger.warning(f"集合维度不匹配: 期望 {self.dim}, 实际 {dim}")
                                    # 删除并重新创建
                                    self.client.drop_collection(self.collection_name)
                                    return self._create_new_collection()
                except Exception as e:
                    logger.warning(f"获取集合信息失败: {e}")
                
                return True
            
            # 创建新集合
            return self._create_new_collection()
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def _create_new_collection(self):
        """创建新集合"""
        try:
            # 使用 MilvusClient 的简化方式创建集合
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dim,
                metric_type="L2"
            )
            
            logger.info(f"集合 '{self.collection_name}' 创建成功 (维度: {self.dim})")
            return True
            
        except Exception as e:
            logger.error(f"创建新集合失败: {e}")
            return False
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本向量"""
        if isinstance(text, list):
            embeddings = self.embedding_model.encode(text, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]
        else:
            embedding = self.embedding_model.encode([text], convert_to_numpy=True)[0]
            return embedding.tolist()
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """添加文档 - 修复字段名问题"""
        try:
            data = []
            for doc in documents:
                text = doc["text"]
                embedding = self.get_embedding(text)
                
                # 准备插入数据 - 使用正确的字段名
                # MilvusClient 需要 'vector' 字段，但可能还需要其他字段
                item = {
                    "vector": embedding,
                    "text": text,
                    "source": doc.get("source", "unknown")
                }
                
                # 添加metadata（如果需要）
                metadata = doc.get("metadata", {})
                if metadata:
                    # 将metadata的键值对添加到item中
                    for key, value in metadata.items():
                        item[key] = str(value)
                
                data.append(item)
            
            # 插入数据
            result = self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            
            logger.info(f"插入了 {len(documents)} 个文档, ID 数量: {len(result.get('ids', []))}")
            return True
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            # 尝试替代方法
            try:
                return self._add_documents_alternative(documents)
            except Exception as e2:
                logger.error(f"替代方法也失败: {e2}")
                raise
    
    def _add_documents_alternative(self, documents: List[Dict[str, Any]]):
        """替代的添加文档方法"""
        try:
            # 更简单的方法：只插入必要的字段
            data = []
            for doc in documents:
                text = doc["text"]
                embedding = self.get_embedding(text)
                
                # 只使用vector字段
                data.append({
                    "vector": embedding
                })
            
            result = self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            
            logger.info(f"使用替代方法插入了 {len(documents)} 个文档")
            return True
            
        except Exception as e:
            logger.error(f"替代方法失败: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        try:
            # 生成查询向量
            query_embedding = self.get_embedding(query)
            
            # 执行搜索
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                search_params={"metric_type": "L2", "params": {}},
                output_fields=["text", "source"]  # 指定要返回的字段
            )
            
            # 格式化结果
            formatted_results = []
            if results and len(results) > 0:
                for result in results[0]:  # results[0] 是因为我们只搜索了一个向量
                    formatted_results.append({
                        "id": result.get("id"),
                        "text": result.get("entity", {}).get("text", ""),
                        "source": result.get("entity", {}).get("source", "unknown"),
                        "score": result.get("distance", 0)
                    })
            
            logger.info(f"搜索到 {len(formatted_results)} 个结果")
            return formatted_results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def delete_all(self):
        """删除所有文档"""
        try:
            # 删除集合
            self.client.drop_collection(self.collection_name)
            logger.info(f"集合 '{self.collection_name}' 已删除")
            
            # 重新创建集合
            self._create_new_collection()
            return True
            
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False
    
    def get_collection_info(self):
        """获取集合信息"""
        try:
            info = self.client.describe_collection(self.collection_name)
            stats = self.client.get_collection_stats(self.collection_name)
            return {
                "description": str(info),
                "stats": stats
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return None