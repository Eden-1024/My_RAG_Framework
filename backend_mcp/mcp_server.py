# mcp_server.py
import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional
import mcp
import mcp.server.stdio
import mcp.types as types
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions

from config import Config
from simple_rag import SimpleRAGSystem

# 配置日志到stderr，避免污染stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # 重要：将日志输出到stderr
)
logger = logging.getLogger(__name__)

class MCPRAGServer:
    """MCP RAG 服务器 - 修复日志输出"""
    
    def __init__(self, config: Config):
        self.config = config
        self.rag_system = SimpleRAGSystem(config)
        self.server = mcp.server.Server("mcp-rag-server")
        
        # 初始化RAG系统
        self.rag_system.initialize()
        
        # 注册工具
        self._register_tools()
        
        # 注册提示模板
        self._register_prompts()
        
        # 注册资源
        self._register_resources()
        
        # 在stderr输出初始化信息
        print("🚀 MCP RAG 服务器启动中...", file=sys.stderr)
        print("📚 已集成的功能：", file=sys.stderr)
        print("   • 向量搜索 (Milvus-Lite)", file=sys.stderr)
        print("   • 知识库管理", file=sys.stderr)
        print("   • RAG问答", file=sys.stderr)
        print("   • MCP协议工具", file=sys.stderr)
        print("\n⚡ 服务器已就绪，等待连接...", file=sys.stderr)
    
    def _register_tools(self):
        """注册MCP工具"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="search_knowledge",
                    description="从知识库中搜索相关信息",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索查询内容"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "返回结果数量，默认5",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="add_to_knowledge",
                    description="添加新知识到知识库",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "要添加的文本内容"
                            },
                            "source": {
                                "type": "string",
                                "description": "来源说明",
                                "default": "user_input"
                            },
                            "category": {
                                "type": "string",
                                "description": "分类标签"
                            }
                        },
                        "required": ["text"]
                    }
                ),
                types.Tool(
                    name="clear_knowledge",
                    description="清空所有知识",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="rag_query",
                    description="使用RAG系统回答问题",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "用户的问题"
                            },
                            "include_context": {
                                "type": "boolean",
                                "description": "是否包含上下文来源",
                                "default": True
                            }
                        },
                        "required": ["question"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(
            name: str,
            arguments: Optional[Dict[str, Any]] = None
        ) -> List[types.TextContent]:
            arguments = arguments or {}
            
            try:
                if name == "search_knowledge":
                    query = arguments.get("query", "")
                    top_k = arguments.get("top_k", 5)
                    
                    logger.info(f"搜索查询: {query}, top_k: {top_k}")
                    results = self.rag_system.search(query, top_k)
                    
                    if not results:
                        return [types.TextContent(
                            type="text",
                            text="没有找到相关信息。"
                        )]
                    
                    output = f"找到 {len(results)} 个相关结果：\n\n"
                    for i, result in enumerate(results, 1):
                        score = result.get('score', 0)
                        similarity = max(0, 1 - score)
                        output += f"{i}. **来源**：{result.get('source', '未知')}\n"
                        output += f"   **相似度**：{similarity:.2%}\n"
                        output += f"   **内容**：{result.get('text', '')}\n\n"
                    
                    return [types.TextContent(type="text", text=output)]
                
                elif name == "add_to_knowledge":
                    text = arguments.get("text", "")
                    source = arguments.get("source", "user_input")
                    category = arguments.get("category", "")
                    
                    logger.info(f"添加文档: 来源={source}")
                    document = {
                        "text": text,
                        "source": source,
                        "metadata": {"category": category} if category else {}
                    }
                    
                    success = self.rag_system.add_documents([document])
                    
                    if success:
                        return [types.TextContent(
                            type="text",
                            text=f"✅ 成功添加到知识库！\n来源：{source}"
                        )]
                    else:
                        return [types.TextContent(
                            type="text",
                            text="❌ 添加失败，请检查日志。"
                        )]
                
                elif name == "clear_knowledge":
                    logger.info("清空知识库")
                    self.rag_system.vector_store.delete_all()
                    self.rag_system.vector_store.create_collection()
                    
                    return [types.TextContent(
                        type="text",
                        text="✅ 知识库已清空！"
                    )]
                
                elif name == "rag_query":
                    question = arguments.get("question", "")
                    include_context = arguments.get("include_context", True)
                    
                    logger.info(f"RAG查询: {question}")
                    
                    if include_context:
                        answer = self.rag_system.query_with_context(question)
                    else:
                        results = self.rag_system.search(question, top_k=3)
                        if results:
                            answer = results[0].get('text', '没有相关信息')
                        else:
                            answer = "没有找到相关信息。"
                    
                    return [types.TextContent(type="text", text=answer)]
                
                else:
                    return [types.TextContent(
                        type="text",
                        text=f"未知工具：{name}"
                    )]
                    
            except Exception as e:
                logger.error(f"工具执行出错: {e}")
                return [types.TextContent(
                    type="text",
                    text=f"工具执行出错：{str(e)}"
                )]
    
    def _register_prompts(self):
        """注册MCP提示模板"""
        
        @self.server.list_prompts()
        async def handle_list_prompts() -> List[types.Prompt]:
            return [
                types.Prompt(
                    name="rag_question",
                    description="使用RAG系统回答问题的提示模板",
                    arguments=[
                        types.PromptArgument(
                            name="question",
                            description="用户的问题",
                            required=True
                        )
                    ]
                ),
                types.Prompt(
                    name="summarize_knowledge",
                    description="总结知识库内容的提示模板",
                    arguments=[
                        types.PromptArgument(
                            name="topic",
                            description="要总结的主题",
                            required=False
                        )
                    ]
                )
            ]
        
        @self.server.get_prompt()
        async def handle_get_prompt(
            name: str,
            arguments: Optional[Dict[str, Any]] = None
        ) -> types.GetPromptResult:
            arguments = arguments or {}
            
            if name == "rag_question":
                question = arguments.get("question", "")
                
                results = self.rag_system.search(question, top_k=3)
                
                messages = []
                
                if results:
                    context = "相关背景知识：\n\n"
                    for i, result in enumerate(results, 1):
                        context += f"{i}. {result.get('text', '')}\n\n"
                    
                    messages.append(
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text=f"{context}\n基于以上信息，请回答这个问题：{question}"
                            )
                        )
                    )
                else:
                    messages.append(
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text=f"问题：{question}\n\n（注：知识库中没有找到相关信息）"
                            )
                        )
                    )
                
                return types.GetPromptResult(
                    messages=messages,
                    description="RAG问题回答提示"
                )
            
            elif name == "summarize_knowledge":
                topic = arguments.get("topic", "")
                
                if topic:
                    results = self.rag_system.search(topic, top_k=10)
                else:
                    results = self.rag_system.search("", top_k=10)
                
                if results:
                    content = f"关于'{topic}'的知识总结：\n\n" if topic else "知识库内容总结：\n\n"
                    
                    sources = {}
                    for result in results:
                        source = result.get('source', '未知')
                        if source not in sources:
                            sources[source] = []
                        sources[source].append(result.get('text', ''))
                    
                    for source, texts in sources.items():
                        content += f"## {source}\n"
                        for text in texts[:3]:
                            content += f"- {text[:100]}...\n"
                        content += "\n"
                    
                    messages = [
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text=f"{content}\n请基于以上知识进行总结："
                            )
                        )
                    ]
                else:
                    messages = [
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text="知识库为空，无法进行总结。"
                            )
                        )
                    ]
                
                return types.GetPromptResult(
                    messages=messages,
                    description="知识总结提示"
                )
            
            else:
                raise ValueError(f"未知提示：{name}")
    
    def _register_resources(self):
        """注册MCP资源"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[types.Resource]:
            return [
                types.Resource(
                    uri="rag://knowledge/stats",
                    name="知识库统计",
                    description="RAG知识库的统计信息",
                    mimeType="application/json"
                ),
                types.Resource(
                    uri="rag://knowledge/sources",
                    name="知识来源",
                    description="知识库中所有文档的来源统计",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            if uri == "rag://knowledge/stats":
                stats = self.rag_system.get_stats()
                return json.dumps(stats, ensure_ascii=False, indent=2)
            
            elif uri == "rag://knowledge/sources":
                results = self.rag_system.search("", top_k=100)
                
                source_count = {}
                for result in results:
                    source = result.get('source', '未知')
                    source_count[source] = source_count.get(source, 0) + 1
                
                return json.dumps({
                    "sources": source_count,
                    "total_documents": len(results)
                }, ensure_ascii=False, indent=2)
            
            else:
                raise ValueError(f"未知资源：{uri}")
    
    async def run(self):
        """运行MCP服务器"""
        try:
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                capabilities = self.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
                
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="mcp-rag-server",
                        server_version="1.0.0",
                        capabilities=capabilities
                    )
                )
        except KeyboardInterrupt:
            print("\n🛑 服务器已停止", file=sys.stderr)
        except Exception as e:
            print(f"❌ 服务器错误: {e}", file=sys.stderr)

async def main():
    """主函数"""
    config = Config()
    server = MCPRAGServer(config)
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())