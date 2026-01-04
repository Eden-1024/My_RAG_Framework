from datetime import datetime
import logging
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, asdict
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ===================== 数据模型定义 =====================
@dataclass
class PageData:
    """页面数据模型"""
    page: int
    text: str

@dataclass
class ChunkMetadata:
    """分块元数据模型"""
    chunk_id: int
    page_number: int
    page_range: str
    word_count: int

@dataclass
class Chunk:
    """分块数据模型"""
    content: str
    metadata: ChunkMetadata

@dataclass
class ChunkingResult:
    """分块结果模型"""
    filename: str
    total_chunks: int
    total_pages: int
    loading_method: str
    chunking_method: str
    timestamp: str
    chunks: List[Chunk]

# ===================== 分块策略接口 =====================
class ChunkingStrategy:
    """分块策略基类"""
    
    def chunk_page(self, page_data: PageData, start_chunk_id: int) -> List[Chunk]:
        """
        分块单个页面
        
        Args:
            page_data: 页面数据
            start_chunk_id: 起始块ID
            
        Returns:
            分块列表
        """
        raise NotImplementedError

class PageChunkingStrategy(ChunkingStrategy):
    """按页面分块策略"""
    
    def chunk_page(self, page_data: PageData, start_chunk_id: int) -> List[Chunk]:
        chunks = []
        metadata = ChunkMetadata(
            chunk_id=start_chunk_id,
            page_number=page_data.page,
            page_range=str(page_data.page),
            word_count=len(page_data.text.split())
        )
        chunks.append(Chunk(content=page_data.text, metadata=metadata))
        return chunks

class FixedSizeChunkingStrategy(ChunkingStrategy):
    """固定大小分块策略"""
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
    
    def chunk_page(self, page_data: PageData, start_chunk_id: int) -> List[Chunk]:
        chunks = []
        page_chunks = self._create_fixed_size_chunks(page_data.text)
        
        for idx, chunk_text in enumerate(page_chunks, 1):
            metadata = ChunkMetadata(
                chunk_id=start_chunk_id + len(chunks),
                page_number=page_data.page,
                page_range=str(page_data.page),
                word_count=len(chunk_text.split())
            )
            chunks.append(Chunk(content=chunk_text, metadata=metadata))
        
        return chunks
    
    def _create_fixed_size_chunks(self, text: str) -> List[str]:
        """创建固定大小的文本块"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + (1 if current_length > 0 else 0)
            
            if current_length + word_length > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(word)
            current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

class ParagraphChunkingStrategy(ChunkingStrategy):
    """按段落分块策略"""
    
    def chunk_page(self, page_data: PageData, start_chunk_id: int) -> List[Chunk]:
        chunks = []
        paragraphs = [p.strip() for p in page_data.text.split('\n\n') if p.strip()]
        
        for para in paragraphs:
            metadata = ChunkMetadata(
                chunk_id=start_chunk_id + len(chunks),
                page_number=page_data.page,
                page_range=str(page_data.page),
                word_count=len(para.split())
            )
            chunks.append(Chunk(content=para, metadata=metadata))
        
        return chunks

class SentenceChunkingStrategy(ChunkingStrategy):
    """按句子分块策略"""
    
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[".", "!", "?", "\n", " "]
        )
    
    def chunk_page(self, page_data: PageData, start_chunk_id: int) -> List[Chunk]:
        chunks = []
        sentences = self.splitter.split_text(page_data.text)
        
        for sentence in sentences:
            metadata = ChunkMetadata(
                chunk_id=start_chunk_id + len(chunks),
                page_number=page_data.page,
                page_range=str(page_data.page),
                word_count=len(sentence.split())
            )
            chunks.append(Chunk(content=sentence, metadata=metadata))
        
        return chunks

# ===================== 策略工厂 =====================
class ChunkingStrategyFactory:
    """分块策略工厂"""
    
    @staticmethod
    def create_strategy(
        method: Literal["by_pages", "fixed_size", "by_paragraphs", "by_sentences"],
        **kwargs
    ) -> ChunkingStrategy:
        """创建分块策略实例"""
        
        strategies = {
            "by_pages": PageChunkingStrategy,
            "fixed_size": FixedSizeChunkingStrategy,
            "by_paragraphs": ParagraphChunkingStrategy,
            "by_sentences": SentenceChunkingStrategy,
        }
        
        if method not in strategies:
            raise ValueError(f"Unsupported chunking method: {method}")
        
        strategy_class = strategies[method]
        return strategy_class(**kwargs) if kwargs else strategy_class()

# ===================== 主服务类 =====================
class ChunkingService:
    """
    文本分块服务，提供多种文本分块策略
    """
    
    def __init__(self):
        self.strategy_factory = ChunkingStrategyFactory()
    
    def chunk_text(
        self,
        text: str,
        method: Literal["by_pages", "fixed_size", "by_paragraphs", "by_sentences"],
        metadata: Dict[str, Any],
        page_map: List[Dict[str, Any]],
        chunk_size: int = 1000
    ) -> Dict[str, Any]:
        """
        将文本按指定方法分块
        
        Args:
            text: 原始文本内容
            method: 分块方法
            metadata: 文档元数据
            page_map: 页面映射列表
            chunk_size: 固定大小分块时的块大小
            
        Returns:
            包含分块结果的文档数据结构
        """
        try:
            # 验证输入
            self._validate_input(page_map, method)
            
            # 转换页面数据为模型对象
            page_data_list = [
                PageData(page=item['page'], text=item['text'])
                for item in page_map
            ]
            
            # 获取分块策略
            strategy_kwargs = {"chunk_size": chunk_size} if method == "fixed_size" else {}
            strategy = self.strategy_factory.create_strategy(method, **strategy_kwargs)
            
            # 执行分块
            chunks = self._execute_chunking(strategy, page_data_list)
            
            # 构建结果
            result = self._build_result(chunks, metadata, method, page_data_list)
            
            logger.info(
                f"Successfully chunked document '{metadata.get('filename', 'unknown')}' "
                f"using method '{method}': {len(chunks)} chunks from {len(page_data_list)} pages"
            )
            
            return asdict(result)
            
        except Exception as e:
            logger.error(f"Error in chunk_text: {str(e)}", exc_info=True)
            raise
    
    def _validate_input(self, page_map: List[Dict[str, Any]], method: str) -> None:
        """验证输入参数"""
        if not page_map:
            raise ValueError("Page map cannot be empty")
        
        for item in page_map:
            if 'page' not in item or 'text' not in item:
                raise ValueError("Page map items must contain 'page' and 'text' keys")
    
    def _execute_chunking(
        self,
        strategy: ChunkingStrategy,
        page_data_list: List[PageData]
    ) -> List[Chunk]:
        """执行分块操作"""
        chunks = []
        chunk_id_counter = 1
        
        for page_data in page_data_list:
            page_chunks = strategy.chunk_page(page_data, chunk_id_counter)
            chunks.extend(page_chunks)
            chunk_id_counter += len(page_chunks)
        
        return chunks
    
    def _build_result(
        self,
        chunks: List[Chunk],
        metadata: Dict[str, Any],
        method: str,
        page_data_list: List[PageData]
    ) -> ChunkingResult:
        """构建分块结果"""
        return ChunkingResult(
            filename=metadata.get("filename", "unknown"),
            total_chunks=len(chunks),
            total_pages=len(page_data_list),
            loading_method=metadata.get("loading_method", "unknown"),
            chunking_method=method,
            timestamp=datetime.now().isoformat(),
            chunks=chunks
        )

# ===================== 使用示例 =====================
def example_usage():
    """使用示例"""
    # 创建服务实例
    service = ChunkingService()
    
    # 准备数据
    text = "这是一个示例文档内容..."
    metadata = {
        "filename": "example.pdf",
        "loading_method": "pdf_loader"
    }
    page_map = [
        {"page": 1, "text": "第一页的内容..."},
        {"page": 2, "text": "第二页的内容..."},
    ]
    
    try:
        # 使用不同的分块方法
        result_by_pages = service.chunk_text(
            text=text,
            method="by_pages",
            metadata=metadata,
            page_map=page_map
        )
        
        result_fixed_size = service.chunk_text(
            text=text,
            method="fixed_size",
            metadata=metadata,
            page_map=page_map,
            chunk_size=500
        )
        
        print(f"按页面分块结果: {result_by_pages['total_chunks']} 个块")
        print(f"固定大小分块结果: {result_fixed_size['total_chunks']} 个块")
        
    except Exception as e:
        logger.error(f"示例执行失败: {e}")

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    example_usage()