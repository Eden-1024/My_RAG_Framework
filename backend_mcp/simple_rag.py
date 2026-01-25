# simple_rag_fixed.py
import asyncio
import logging
from typing import List, Dict, Any
from config import Config
from milvus_manager import MilvusLiteManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleRAGSystem:
    """简化的 RAG 系统 """
    
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
            self.vector_store.add_documents(documents)
            logger.info(f"成功添加 {len(documents)} 个文档")
            return True
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

def main():
    """主函数 - 测试 RAG 系统"""
    
    # 创建配置
    config = Config()
    
    # 创建 RAG 系统
    rag = SimpleRAGSystem(config)
    
    try:
        # 初始化
        if not rag.initialize():
            print("✗ RAG 系统初始化失败！")
            return
        
        print("✓ RAG 系统初始化成功！")
        
        # 添加示例文档
        documents = [
            {
                "text": "MCP (Model Context Protocol) 是一个用于在 AI 应用程序和工具之间交换上下文信息的协议。它允许不同的 AI 工具共享和访问上下文数据。",
                "source": "mcp_documentation",
                "metadata": {"category": "protocol", "version": "1.26.0"}
            },
            {
                "text": "Milvus 是一个开源的向量数据库，专门用于存储和检索大规模向量数据。它支持相似度搜索、聚类和分类等操作。Milvus-Lite 是 Milvus 的轻量级版本。",
                "source": "milvus_docs",
                "metadata": {"category": "database", "version": "2.4.9"}
            },
            {
                "text": "RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的 AI 技术。它通过检索相关知识来增强生成模型的能力，提高回答的准确性和相关性。",
                "source": "ai_concepts",
                "metadata": {"category": "technique"}
            },
            {
                "text": "向量嵌入 (Vector Embeddings) 是将文本、图像或其他数据转换为数值向量的过程。这些向量可以捕获数据的语义信息，用于相似度计算。",
                "source": "ai_concepts",
                "metadata": {"category": "technique"}
            },
            {
                "text": "Python 是一种广泛使用的高级编程语言，以其简洁的语法和强大的库生态系统而闻名。常用于机器学习和数据科学。",
                "source": "programming",
                "metadata": {"category": "language"}
            }
        ]
        
        if rag.add_documents(documents):
            print(f"✓ 成功添加 {len(documents)} 个示例文档")
        
        # 获取统计信息
        stats = rag.get_stats()
        if stats:
            print(f"\n集合信息：")
            print(f"  文档数量: {stats.get('stats', {}).get('row_count', '未知')}")
        
        # 测试查询
        test_queries = [
            "什么是 MCP？",
            "Milvus 是什么数据库？",
            "解释一下 RAG 技术",
            "向量嵌入是什么意思？",
            "Python 有什么特点？"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"查询: {query}")
            print(f"{'='*60}")
            
            # 搜索相关文档
            results = rag.search(query, top_k=2)
            
            if results:
                print(f"找到 {len(results)} 个相关结果:")
                for i, result in enumerate(results, 1):
                    score = result.get('score', 0)
                    similarity = max(0, 1 - score)
                    print(f"\n结果 {i}:")
                    print(f"  来源: {result.get('source', 'unknown')}")
                    print(f"  相似度: {similarity:.2%}")
                    print(f"  内容: {result.get('text', '')[:120]}...")
            else:
                print("没有找到相关结果")
        
        # 测试带上下文的查询
        print(f"\n{'='*60}")
        print("完整 RAG 响应示例:")
        print(f"{'='*60}")
        response = rag.query_with_context("请解释 MCP 和 RAG 的关系")
        print(response)
        
        print(f"\n{'='*60}")
        print("RAG 系统测试完成！")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()