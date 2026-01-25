# interactive_client_final_fixed.py
import asyncio
import json
import sys
import os
from typing import Optional, Dict, Any
import mcp
import mcp.client.stdio
import mcp.client.session
from mcp.client.stdio import StdioServerParameters

class MCPRAGClient:
    """MCP RAG客户端 - 最终修复版"""
    
    def __init__(self, server_command: str, server_args: list):
        self.server_params = StdioServerParameters(
            command=server_command,
            args=server_args
        )
    
    async def run_interactive(self):
        """运行交互式客户端"""
        print("=" * 60)
        print("MCP RAG 交互式客户端")
        print("=" * 60)
        
        try:
            async with mcp.client.stdio.stdio_client(self.server_params) as (read_stream, write_stream):
                async with mcp.client.session.ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    print("\n✅ 服务器已连接！")
                    print("=" * 60)
                    
                    while True:
                        print("\n命令菜单:")
                        print("  1. 列出所有工具")
                        print("  2. 搜索知识库")
                        print("  3. 添加新知识")
                        print("  4. 提问")
                        print("  5. 查看统计信息")
                        print("  6. 列出提示模板")
                        print("  7. 列出所有资源")
                        print("  8. 测试所有功能")
                        print("  9. 退出")
                        
                        choice = input("\n请选择 (1-9): ").strip()
                        
                        if choice == "9":
                            print("再见！")
                            break
                        
                        elif choice == "1":
                            await self._handle_list_tools(session)
                        
                        elif choice == "2":
                            await self._handle_search(session)
                        
                        elif choice == "3":
                            await self._handle_add_knowledge(session)
                        
                        elif choice == "4":
                            await self._handle_ask_question(session)
                        
                        elif choice == "5":
                            await self._handle_show_stats(session)
                        
                        elif choice == "6":
                            await self._handle_list_prompts(session)
                        
                        elif choice == "7":
                            await self._handle_list_resources(session)
                        
                        elif choice == "8":
                            await self._test_all_features(session)
                        
                        else:
                            print("无效选择，请重试")
                            
        except Exception as e:
            print(f"❌ 客户端错误: {e}")
            import traceback
            traceback.print_exc()
    
    async def _handle_list_tools(self, session):
        """处理列出工具"""
        try:
            print("\n获取工具列表...")
            tools_result = await session.list_tools()
            if hasattr(tools_result, 'tools'):
                tools = tools_result.tools
                print(f"\n🛠️ 可用工具 ({len(tools)}):")
                for i, tool in enumerate(tools, 1):
                    print(f"\n  {i}. {tool.name}")
                    print(f"     描述: {tool.description}")
            else:
                print("工具列表格式错误")
        except Exception as e:
            print(f"列出工具失败: {e}")
    
    async def _handle_search(self, session):
        """处理搜索"""
        query = input("\n请输入搜索内容: ").strip()
        if not query:
            print("搜索内容不能为空")
            return
        
        try:
            top_k = input("返回结果数量 (默认3): ").strip()
            top_k = int(top_k) if top_k else 3
            
            print(f"\n🔍 搜索: '{query}' (返回 {top_k} 个结果)")
            
            result = await session.call_tool(
                "search_knowledge",
                {"query": query, "top_k": top_k}
            )
            
            # 处理CallToolResult - 使用content属性（单数）
            if hasattr(result, 'content'):
                contents = result.content
                if contents:
                    for content in contents:
                        if hasattr(content, 'text'):
                            print(f"\n{content.text}")
                else:
                    print("没有找到相关结果")
            else:
                print(f"搜索结果格式错误，返回类型: {type(result)}")
                
        except ValueError:
            print("请输入有效的数字")
        except Exception as e:
            print(f"搜索失败: {e}")
    
    async def _handle_add_knowledge(self, session):
        """处理添加知识"""
        print("\n📝 添加新知识到知识库")
        text = input("请输入文本: ").strip()
        if not text:
            print("文本不能为空")
            return
        
        source = input("来源 (可选): ").strip() or "user_input"
        category = input("分类 (可选): ").strip() or ""
        
        try:
            arguments = {"text": text, "source": source}
            if category:
                arguments["category"] = category
            
            print(f"\n添加中...")
            print(f"  文本: {text[:100]}...")
            print(f"  来源: {source}")
            if category:
                print(f"  分类: {category}")
            
            result = await session.call_tool("add_to_knowledge", arguments)
            
            # 处理CallToolResult - 使用content属性（单数）
            if hasattr(result, 'content'):
                contents = result.content
                if contents:
                    for content in contents:
                        if hasattr(content, 'text'):
                            print(f"\n✅ {content.text}")
                else:
                    print("添加失败，无返回结果")
            else:
                print(f"添加结果格式错误，返回类型: {type(result)}")
                
        except Exception as e:
            print(f"添加失败: {e}")
    
    async def _handle_ask_question(self, session):
        """处理提问"""
        question = input("\n请输入问题: ").strip()
        if not question:
            print("问题不能为空")
            return
        
        try:
            include_context = input("包含上下文来源? (y/n, 默认y): ").strip().lower()
            include_context = include_context != 'n'
            
            print(f"\n🤖 正在回答问题: '{question}'")
            if include_context:
                print("  包含上下文来源")
            
            result = await session.call_tool(
                "rag_query",
                {
                    "question": question,
                    "include_context": include_context
                }
            )
            
            # 处理CallToolResult - 使用content属性（单数）
            if hasattr(result, 'content'):
                contents = result.content
                if contents:
                    for content in contents:
                        if hasattr(content, 'text'):
                            print(f"\n{content.text}")
                else:
                    print("无法回答问题")
            else:
                print(f"回答结果格式错误，返回类型: {type(result)}")
                
        except Exception as e:
            print(f"提问失败: {e}")
    
    async def _handle_show_stats(self, session):
        """处理显示统计"""
        try:
            print("\n📊 获取知识库统计...")
            
            resources_result = await session.list_resources()
            if not hasattr(resources_result, 'resources'):
                print("资源列表格式错误")
                return
            
            resources = resources_result.resources
            
            # 查找统计资源
            stats_uri = None
            for resource in resources:
                if "stats" in resource.name.lower() or "统计" in resource.name:
                    stats_uri = resource.uri
                    break
            
            if not stats_uri:
                stats_uri = "rag://knowledge/stats"
            
            print(f"读取资源: {stats_uri}")
            
            result = await session.read_resource(stats_uri)
            
            if hasattr(result, 'contents'):
                contents = result.contents
                if contents:
                    for content in contents:
                        if hasattr(content, 'text'):
                            try:
                                stats = json.loads(content.text)
                                print(json.dumps(stats, indent=2, ensure_ascii=False))
                            except json.JSONDecodeError:
                                print(content.text)
                else:
                    print("未找到统计信息")
            else:
                print("统计结果格式错误")
                
        except Exception as e:
            print(f"获取统计失败: {e}")
    
    async def _handle_list_prompts(self, session):
        """处理列出提示"""
        try:
            print("\n获取提示模板列表...")
            prompts_result = await session.list_prompts()
            if hasattr(prompts_result, 'prompts'):
                prompts = prompts_result.prompts
                print(f"\n💡 提示模板 ({len(prompts)}):")
                for i, prompt in enumerate(prompts, 1):
                    print(f"\n  {i}. {prompt.name}")
                    print(f"     描述: {prompt.description}")
            else:
                print("提示列表格式错误")
        except Exception as e:
            print(f"列出提示失败: {e}")
    
    async def _handle_list_resources(self, session):
        """处理列出资源"""
        try:
            print("\n获取资源列表...")
            resources_result = await session.list_resources()
            if hasattr(resources_result, 'resources'):
                resources = resources_result.resources
                print(f"\n📚 资源 ({len(resources)}):")
                for i, resource in enumerate(resources, 1):
                    print(f"\n  {i}. {resource.name}")
                    print(f"     描述: {resource.description}")
                    print(f"     URI: {resource.uri}")
            else:
                print("资源列表格式错误")
        except Exception as e:
            print(f"列出资源失败: {e}")
    
    async def _test_all_features(self, session):
        """测试所有功能"""
        print("\n" + "="*60)
        print("测试所有功能")
        print("="*60)
        
        try:
            # 1. 测试工具
            print("\n1. 测试工具功能...")
            tools_result = await session.list_tools()
            if hasattr(tools_result, 'tools'):
                tools = tools_result.tools
                print(f"   找到 {len(tools)} 个工具")
            
            # 2. 测试搜索
            print("\n2. 测试搜索功能...")
            search_result = await session.call_tool(
                "search_knowledge",
                {"query": "测试", "top_k": 2}
            )
            if hasattr(search_result, 'content'):
                contents = search_result.content
                print(f"   搜索完成，返回 {len(contents)} 个结果")
            
            # 3. 测试添加
            print("\n3. 测试添加功能...")
            add_result = await session.call_tool(
                "add_to_knowledge",
                {
                    "text": "这是功能测试添加的文档内容。",
                    "source": "function_test",
                    "category": "test"
                }
            )
            if hasattr(add_result, 'content'):
                contents = add_result.content
                if contents:
                    print(f"   添加完成: {contents[0].text}")
            
            # 4. 测试提问
            print("\n4. 测试提问功能...")
            ask_result = await session.call_tool(
                "rag_query",
                {"question": "什么是测试?", "include_context": False}
            )
            if hasattr(ask_result, 'content'):
                contents = ask_result.content
                if contents:
                    print(f"   提问完成，回答长度: {len(contents[0].text)}")
            
            # 5. 测试提示
            print("\n5. 测试提示功能...")
            prompts_result = await session.list_prompts()
            if hasattr(prompts_result, 'prompts'):
                prompts = prompts_result.prompts
                print(f"   找到 {len(prompts)} 个提示模板")
            
            # 6. 测试资源
            print("\n6. 测试资源功能...")
            resources_result = await session.list_resources()
            if hasattr(resources_result, 'resources'):
                resources = resources_result.resources
                print(f"   找到 {len(resources)} 个资源")
            
            print("\n" + "="*60)
            print("✅ 所有功能测试完成！")
            print("="*60)
            
        except Exception as e:
            print(f"❌ 功能测试失败: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """主函数"""
    print("MCP RAG 交互式客户端 (最终修复版)")
    print("=" * 60)
    
    # 获取当前Python解释器和脚本路径
    python_exe = sys.executable
    server_script = os.path.abspath("mcp_server.py")
    
    print(f"Python解释器: {python_exe}")
    print(f"服务器脚本: {server_script}")
    
    # 检查服务器脚本是否存在
    if not os.path.exists(server_script):
        print(f"\n❌ 服务器脚本不存在: {server_script}")
        print("请确保以下文件存在:")
        print("  1. mcp_server.py")
        print("  2. config.py")
        print("  3. milvus_manager.py")
        print("  4. simple_rag.py")
        return
    
    print("\n正在启动客户端...")
    
    client = MCPRAGClient(
        server_command=python_exe,
        server_args=[server_script]
    )
    
    try:
        await client.run_interactive()
    except KeyboardInterrupt:
        print("\n\n客户端已停止")
    except Exception as e:
        print(f"\n❌ 客户端运行失败: {e}")

if __name__ == "__main__":
    # 设置Windows上的asyncio事件循环策略
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已停止")