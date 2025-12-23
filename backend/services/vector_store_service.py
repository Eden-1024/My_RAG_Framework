import os
from datetime import datetime
import json
from typing import List, Dict, Any
import logging
from pathlib import Path
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except Exception:
    chromadb = None
    Settings = None
    CHROMA_AVAILABLE = False
from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from utils.config import VectorDBProvider, MILVUS_CONFIG  # Updated import
from pypinyin import lazy_pinyin, Style

logger = logging.getLogger(__name__)

class VectorDBConfig:
    """
    向量数据库配置类，用于存储和管理向量数据库的配置信息
    """
    def __init__(self, provider: str, index_mode: str):
        """
        初始化向量数据库配置
        
        参数:
            provider: 向量数据库提供商名称
            index_mode: 索引模式
        """
        self.provider = provider
        self.index_mode = index_mode
        self.milvus_uri = MILVUS_CONFIG["uri"]

    def _get_milvus_index_type(self, index_mode: str) -> str:
        """
        根据索引模式获取Milvus索引类型
        
        参数:
            index_mode: 索引模式
            
        返回:
            对应的Milvus索引类型
        """
        return MILVUS_CONFIG["index_types"].get(index_mode, "FLAT")
    
    def _get_milvus_index_params(self, index_mode: str) -> Dict[str, Any]:
        """
        根据索引模式获取Milvus索引参数
        
        参数:
            index_mode: 索引模式
            
        返回:
            对应的Milvus索引参数字典
        """
        return MILVUS_CONFIG["index_params"].get(index_mode, {})

class VectorStoreService:
    """
    向量存储服务类，提供向量数据的索引、查询和管理功能
    """
    def __init__(self):
        """
        初始化向量存储服务
        """
        self.initialized_dbs = {}
        # 确保存储目录存在
        os.makedirs("03-vector-store", exist_ok=True)

    def _safe_persist_client(self, client, settings=None):
        """
        安全调用 chromadb client.persist()，兼容没有 persist 方法的旧/新版本 chromadb。
        """
        try:
            persist_fn = getattr(client, "persist", None)
            if callable(persist_fn):
                persist_fn()
            else:
                logger.info("Chroma client does not support persist() method.")
        except Exception as e:
            # 不要阻塞主流程，仅记录异常
            logger.exception("Chroma client.persist() failed for %s: %s", (settings.persist_directory if settings else '<none>'), e)

    def _get_chroma_index_type(self, config: VectorDBConfig) -> str:
        """
        从配置对象获取Chroma索引类型（占位符函数）
        
        参数:
            config: 向量数据库配置对象
            
        返回:
            Chroma索引类型
        """
        # 这里可以根据需要实现Chroma的索引类型获取逻辑
        return "default"
    
    def _get_milvus_index_type(self, config: VectorDBConfig) -> str:
        """
        从配置对象获取Milvus索引类型
        
        参数:
            config: 向量数据库配置对象
            
        返回:
            Milvus索引类型
        """
        return config._get_milvus_index_type(config.index_mode)

    def _get_chroma_index_params(self, config: VectorDBConfig) -> Dict[str, Any]:
        """
        从配置对象获取Chroma索引参数（占位符函数）
        
        参数:
            config: 向量数据库配置对象
            
        返回:
            Chroma索引参数字典
        """
        # 这里可以根据需要实现Chroma的索引参数获取逻辑
        return {}
    
    def _get_milvus_index_params(self, config: VectorDBConfig) -> Dict[str, Any]:
        """
        从配置对象获取Milvus索引参数
        
        参数:
            config: 向量数据库配置对象
            
        返回:
            Milvus索引参数字典
        """
        return config._get_milvus_index_params(config.index_mode)

    def index_embeddings(self, embedding_file: str, config: VectorDBConfig) -> Dict[str, Any]:
        """
        将嵌入向量索引到向量数据库
        
        参数:
            embedding_file: 嵌入向量文件路径
            config: 向量数据库配置对象
            
        返回:
            索引结果信息字典
        """
        start_time = datetime.now()
        
        # 读取embedding文件
        embeddings_data = self._load_embeddings(embedding_file)
        
        # 根据不同的数据库进行索引
        if config.provider == VectorDBProvider.MILVUS:
            result = self._index_to_milvus(embeddings_data, config)
        elif config.provider == VectorDBProvider.CHROMA:
            result = self._index_to_chroma(embeddings_data, config)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return {
            "database": config.provider,
            "index_mode": config.index_mode,
            "total_vectors": len(embeddings_data["embeddings"]),
            "index_size": result.get("index_size", "N/A"),
            "processing_time": processing_time,
            "collection_name": result.get("collection_name", "N/A")
        }
    
    def _load_embeddings(self, file_path: str) -> Dict[str, Any]:
        """
        加载embedding文件，返回配置信息和embeddings
        
        参数:
            file_path: 嵌入向量文件路径
            
        返回:
            包含嵌入向量和元数据的字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loading embeddings from {file_path}")
                
                if not isinstance(data, dict) or "embeddings" not in data:
                    raise ValueError("Invalid embedding file format: missing 'embeddings' key")
                    
                # 返回完整的数据，包括顶层配置
                logger.info(f"Found {len(data['embeddings'])} embeddings")
                return data
                
        except Exception as e:
            logger.error(f"Error loading embeddings from {file_path}: {str(e)}")
            raise
    
    def _index_to_milvus(self, embeddings_data: Dict[str, Any], config: VectorDBConfig) -> Dict[str, Any]:
        """
        将嵌入向量索引到Milvus数据库
        
        参数:
            embeddings_data: 嵌入向量数据
            config: 向量数据库配置对象
            
        返回:
            索引结果信息字典
        """
        try:
            # 使用 filename 作为 collection 名称前缀
            filename = embeddings_data.get("filename", "")
            # 如果有 .pdf 后缀，移除它
            base_name = filename.replace('.pdf', '') if filename else "doc"
            
            # Convert Chinese characters to pinyin
            base_name = ''.join(lazy_pinyin(base_name, style=Style.NORMAL))
            
            # Replace hyphens with underscores in the base name
            base_name = base_name.replace('-', '_')
            
            # Ensure the collection name starts with a letter or underscore
            if not base_name[0].isalpha() and base_name[0] != '_':
                base_name = f"_{base_name}"
            
            # Get embedding provider
            embedding_provider = embeddings_data.get("embedding_provider", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            collection_name = f"{base_name}_{embedding_provider}_{timestamp}"
            
            # 连接到Milvus
            connections.connect(
                alias="default", 
                uri=config.milvus_uri
            )
            
            # 从顶层配置获取向量维度
            vector_dim = int(embeddings_data.get("vector_dimension"))
            if not vector_dim:
                raise ValueError("Missing vector_dimension in embedding file")
            
            logger.info(f"Creating collection with dimension: {vector_dim}")
            
            # 定义字段
            fields = [
                {"name": "id", "dtype": "INT64", "is_primary": True, "auto_id": True},
                {"name": "content", "dtype": "VARCHAR", "max_length": 5000},
                {"name": "document_name", "dtype": "VARCHAR", "max_length": 255},
                {"name": "chunk_id", "dtype": "INT64"},
                {"name": "total_chunks", "dtype": "INT64"},
                {"name": "word_count", "dtype": "INT64"},
                {"name": "page_number", "dtype": "VARCHAR", "max_length": 10},
                {"name": "page_range", "dtype": "VARCHAR", "max_length": 10},
                # {"name": "chunking_method", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_provider", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_model", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_timestamp", "dtype": "VARCHAR", "max_length": 50},
                {
                    "name": "vector",
                    "dtype": "FLOAT_VECTOR",
                    "dim": vector_dim,
                    "params": self._get_milvus_index_params(config)
                }
            ]
            
            # 准备数据为列表格式
            entities = []
            for emb in embeddings_data["embeddings"]:
                entity = {
                    "content": str(emb["metadata"].get("content", "")),
                    "document_name": embeddings_data.get("filename", ""),  # 使用 filename 而不是 document_name
                    "chunk_id": int(emb["metadata"].get("chunk_id", 0)),
                    "total_chunks": int(emb["metadata"].get("total_chunks", 0)),
                    "word_count": int(emb["metadata"].get("word_count", 0)),
                    "page_number": str(emb["metadata"].get("page_number", 0)),
                    "page_range": str(emb["metadata"].get("page_range", "")),
                    # "chunking_method": str(emb["metadata"].get("chunking_method", "")),
                    "embedding_provider": embeddings_data.get("embedding_provider", ""),  # 从顶层配置获取
                    "embedding_model": embeddings_data.get("embedding_model", ""),  # 从顶层配置获取
                    "embedding_timestamp": str(emb["metadata"].get("embedding_timestamp", "")),
                    "vector": [float(x) for x in emb.get("embedding", [])]
                }
                entities.append(entity)
            
            logger.info(f"Creating Milvus collection: {collection_name}")
            
            # 创建collection
            # field_schemas = [
            #     FieldSchema(name=field["name"], 
            #                dtype=getattr(DataType, field["dtype"]),
            #                is_primary="is_primary" in field and field["is_primary"],
            #                auto_id="auto_id" in field and field["auto_id"],
            #                max_length=field.get("max_length"),
            #                dim=field.get("dim"),
            #                params=field.get("params"))
            #     for field in fields
            # ]

            field_schemas = []
            for field in fields:
                extra_params = {}
                if field.get('max_length') is not None:
                    extra_params['max_length'] = field['max_length']
                if field.get('dim') is not None:
                    extra_params['dim'] = field['dim']
                if field.get('params') is not None:
                    extra_params['params'] = field['params']
                field_schema = FieldSchema(
                    name=field["name"], 
                    dtype=getattr(DataType, field["dtype"]),
                    is_primary=field.get("is_primary", False),
                    auto_id=field.get("auto_id", False),
                    **extra_params
                )
                field_schemas.append(field_schema)

            schema = CollectionSchema(fields=field_schemas, description=f"Collection for {collection_name}")
            collection = Collection(name=collection_name, schema=schema)
            
            # 插入数据
            logger.info(f"Inserting {len(entities)} vectors")
            insert_result = collection.insert(entities)
            
            # 创建索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": self._get_milvus_index_type(config),
                "params": self._get_milvus_index_params(config)
            }
            collection.create_index(field_name="vector", index_params=index_params)
            collection.load()
            
            return {
                "index_size": len(insert_result.primary_keys),
                "collection_name": collection_name
            }
            
        except Exception as e:
            logger.error(f"Error indexing to Milvus: {str(e)}")
            raise
        
        finally:
            connections.disconnect("default")

    def _index_to_chroma(self, embeddings_data: Dict[str, Any], config: VectorDBConfig) -> Dict[str, Any]:
        """
        将嵌入向量索引到 Chroma
        
        参数:
            embeddings_data: 嵌入向量数据
            config: 向量数据库配置对象
            
        返回:
            索引结果信息字典
        """
        if not CHROMA_AVAILABLE or chromadb is None:
            raise RuntimeError("chromadb is not available. Please install chromadb to use CHROMA provider.")

        try:
            filename = embeddings_data.get("filename", "")
            base_name = filename.replace('.pdf', '') if filename else "doc"
            base_name = ''.join(lazy_pinyin(base_name, style=Style.NORMAL))
            base_name = base_name.replace('-', '_')
            if not base_name[0].isalpha() and base_name[0] != '_':
                base_name = f"_{base_name}"

            embedding_provider = embeddings_data.get("embedding_provider", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            collection_name = f"{base_name}_{embedding_provider}_{timestamp}"
            print(f"Chroma collection name: {collection_name}")

            # 创建/连接 Chroma 客户端（使用本地目录持久化）
            settings = Settings(persist_directory=str(Path("03-vector-store").resolve())) if Settings is not None else None
            client = chromadb.Client(settings) if settings is not None else chromadb.Client()

            # 准备要插入的数据
            embeddings = []
            metadatas = []
            documents = []
            ids = []
            for idx, emb in enumerate(embeddings_data.get("embeddings", [])):
                ids.append(f"{timestamp}_{idx}")
                vec = [float(x) for x in emb.get("embedding", [])]
                embeddings.append(vec)
                meta = emb.get("metadata", {})
                # Ensure metadata values are JSON-serializable
                safe_meta = {k: (v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v)) for k, v in meta.items()}
                metadatas.append(safe_meta)
                documents.append(str(meta.get("content", "")))

            # 创建集合（如果已存在则获取）
            try:
                collection = client.get_collection(name=collection_name)
            except Exception:
                collection = client.create_collection(name=collection_name)

            # 添加数据
            collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

            # 持久化到磁盘（如果支持）
            # 安全持久化（在某些 chromadb 版本中没有 persist 方法）
            self._safe_persist_client(client, settings)

            return {
                "index_size": len(ids),
                "collection_name": collection_name
            }

        except Exception as e:
            logger.error(f"Error indexing to Chroma: {str(e)}")
            raise

    def list_collections(self, provider: str) -> List[str]:
        """
        列出指定提供商的所有集合
        
        参数:
            provider: 向量数据库提供商
            
        返回:
            集合名称列表
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])
                collections = utility.list_collections()
                return collections
            finally:
                connections.disconnect("default")
        elif provider == VectorDBProvider.CHROMA:
            if not CHROMA_AVAILABLE or chromadb is None:
                raise RuntimeError("chromadb is not available. Please install chromadb to use CHROMA provider.")
            settings = Settings(persist_directory=str(Path("03-vector-store").resolve())) if Settings is not None else None
            client = chromadb.Client(settings) if settings is not None else chromadb.Client()
            try:
                cols = client.list_collections()
                # list_collections may return list of dicts like [{'name': '...'}]
                names = []
                for c in cols:
                    if isinstance(c, dict) and 'name' in c:
                        names.append(c['name'])
                    elif hasattr(c, 'name'):
                        names.append(getattr(c, 'name'))
                    else:
                        # fallback: string
                        names.append(str(c))
                return names
            finally:
                # 安全持久化
                self._safe_persist_client(client, settings)
        return []

    def delete_collection(self, provider: str, collection_name: str) -> bool:
        """
        删除指定的集合
        
        参数:
            provider: 向量数据库提供商
            collection_name: 集合名称
            
        返回:
            是否删除成功
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])
                utility.drop_collection(collection_name)
                return True
            finally:
                connections.disconnect("default")
        elif provider == VectorDBProvider.CHROMA:
            if not CHROMA_AVAILABLE or chromadb is None:
                raise RuntimeError("chromadb is not available. Please install chromadb to use CHROMA provider.")
            settings = Settings(persist_directory=str(Path("03-vector-store").resolve())) if Settings is not None else None
            client = chromadb.Client(settings) if settings is not None else chromadb.Client()
            try:
                # try client.delete_collection if available
                try:
                    client.delete_collection(name=collection_name)
                except TypeError:
                    # older API signature
                    client.delete_collection(collection_name)
                return True
            except Exception as e:
                logger.error(f"Error deleting Chroma collection {collection_name}: {e}")
                return False
        return False

    def get_collection_info(self, provider: str, collection_name: str) -> Dict[str, Any]:
        """
        获取指定集合的信息
        
        参数:
            provider: 向量数据库提供商
            collection_name: 集合名称
            
        返回:
            集合信息字典
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])
                collection = Collection(collection_name)
                return {
                    "name": collection_name,
                    "num_entities": collection.num_entities,
                    "schema": collection.schema.to_dict()
                }
            finally:
                connections.disconnect("default")
        elif provider == VectorDBProvider.CHROMA:
            if not CHROMA_AVAILABLE or chromadb is None:
                raise RuntimeError("chromadb is not available. Please install chromadb to use CHROMA provider.")
            settings = Settings(persist_directory=str(Path("03-vector-store").resolve())) if Settings is not None else None
            client = chromadb.Client(settings) if settings is not None else chromadb.Client()
            try:
                collection = client.get_collection(name=collection_name)
                num = None
                try:
                    num = collection.count()
                except Exception:
                    # fallback: try querying a small batch to estimate
                    try:
                        res = collection.get(ids=[None], include=['metadatas'])
                        num = len(res.get('ids', []))
                    except Exception:
                        num = None
                return {
                    "name": collection_name,
                    "num_entities": num,
                    "schema": None
                }
            finally:
                # 安全持久化
                self._safe_persist_client(client, settings)
        return {}