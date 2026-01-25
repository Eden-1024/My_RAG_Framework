# example.py
import asyncio
from config import Config
from rag_system import MCPRAGSystem

def example_usage():
    """示例使用"""
    
    # 加载配置
    config = Config.from_env()
    
    # 创建 RAG 系统
    rag_system = MCPRAGSystem(config)
    
    try:
        # 初始化
        rag_system.initialize()
        print("RAG 系统初始化成功!")
        
        # 添加示例文档
        documents = [
            {
                "text": "MCP (Model Context Protocol) 是一个用于在 AI 应用程序和工具之间交换上下文信息的协议。",
                "source": "mcp_docs",
                "metadata": {"category": "protocol", "version": "1.26.0"}
            },
            {
                "text": "Milvus 是一个开源的向量数据库，专门用于存储和检索大规模向量数据。",
                "source": "milvus_docs",
                "metadata": {"category": "database", "version": "2.4.9"}
            },
            {
                "text": "RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的 AI 技术。",
                "source": "ai_concepts",
                "metadata": {"category": "technique"}
            }
        ]
        
        rag_system.add_documents(documents)
        print("示例文档添加成功!")
        
        # 测试查询
        query = "什么是 MCP？"
        results = rag_system.query(query, top_k=2)
        
        print(f"\n查询: {query}")
        print(f"找到 {len(results)} 个相关结果:")
        for i, result in enumerate(results, 1):
            print(f"\n结果 {i}:")
            print(f"  相似度: {result['score']:.4f}")
            print(f"  来源: {result['source']}")
            print(f"  内容: {result['text']}")
        
    except Exception as e:
        print(f"错误: {e}")

async def main():
    """主函数"""
    
    # 示例用法
    example_usage()
    
    # 如果要启动 MCP 服务器（需要手动运行）
    # print("\n要启动 MCP 服务器，请运行: python -m mcp_server")

if __name__ == "__main__":
    print("启动示例...")
    asyncio.run(main())