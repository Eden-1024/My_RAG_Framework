# config.py
import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class Config:
    """MCP RAG 系统配置 - 最终版"""
    # MCP 配置
    mcp_host: str = "localhost"
    mcp_port: int = 8000
    
    # Milvus-Lite 配置
    milvus_db_path: str = "./milvus_data.db"
    
    # 向量模型配置
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # 集合配置
    collection_name: str = "mcp_rag_docs"
    
    @classmethod
    def from_env(cls):
        """从环境变量加载配置"""
        return cls(
            mcp_host=os.getenv("MCP_HOST", "localhost"),
            mcp_port=int(os.getenv("MCP_PORT", "8000")),
            milvus_db_path=os.getenv("MILVUS_DB_PATH", "./milvus_data.db"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            collection_name=os.getenv("COLLECTION_NAME", "mcp_rag_docs")
        )