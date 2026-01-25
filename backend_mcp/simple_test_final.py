# test_client_fixed_final.py
import asyncio
import sys
import os
import mcp
import mcp.client.stdio
import mcp.client.session
from mcp.client.stdio import StdioServerParameters

async def test_with_proper_handling():
    """使用正确的处理方式测试"""
    print("\n使用正确的处理方式测试...")
    
    try:
        params = StdioServerParameters(
            command=sys.executable,
            args=[os.path.abspath("mcp_server.py")]
        )
        
        async with mcp.client.stdio.stdio_client(params) as (read_stream, write_stream):
            async with mcp.client.session.ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                print("\n=== 测试结果 ===")
                
                # 工具测试
                try:
                    tools_result = await session.list_tools()
                    if hasattr(tools_result, 'tools'):
                        print(f"✅ 工具: 找到 {len(tools_result.tools)} 个工具")
                        for tool in tools_result.tools[:3]:
                            print(f"   - {tool.name}: {tool.description}")
                    else:
                        print(f"⚠️  工具结果格式: {type(tools_result)}")
                except Exception as e:
                    print(f"❌ 工具测试失败: {e}")
                
                # 提示测试
                try:
                    prompts_result = await session.list_prompts()
                    if hasattr(prompts_result, 'prompts'):
                        print(f"✅ 提示: 找到 {len(prompts_result.prompts)} 个提示")
                        for prompt in prompts_result.prompts:
                            print(f"   - {prompt.name}: {prompt.description}")
                    else:
                        print(f"⚠️  提示结果格式: {type(prompts_result)}")
                except Exception as e:
                    print(f"❌ 提示测试失败: {e}")
                
                # 资源测试
                try:
                    resources_result = await session.list_resources()
                    if hasattr(resources_result, 'resources'):
                        print(f"✅ 资源: 找到 {len(resources_result.resources)} 个资源")
                        for resource in resources_result.resources:
                            print(f"   - {resource.name}: {resource.description}")
                    else:
                        print(f"⚠️  资源结果格式: {type(resources_result)}")
                except Exception as e:
                    print(f"❌ 资源测试失败: {e}")
                
                # 工具调用测试 - 正确处理CallToolResult
                try:
                    print("\n🔧 工具调用测试...")
                    call_result = await session.call_tool(
                        "search_knowledge",
                        {"query": "测试", "top_k": 1}
                    )
                    
                    # 根据API文档，CallToolResult有content属性（注意是单数）
                    if hasattr(call_result, 'content'):
                        content = call_result.content
                        print(f"✅ 工具调用成功，返回 {len(content)} 个内容")
                        
                        if content:
                            for item in content[:1]:
                                if hasattr(item, 'text'):
                                    text_preview = item.text[:100] + "..." if len(item.text) > 100 else item.text
                                    print(f"   结果预览: {text_preview}")
                                elif hasattr(item, 'type'):
                                    print(f"   内容类型: {item.type}")
                    else:
                        print(f"⚠️  CallToolResult格式: {type(call_result)}")
                        print(f"   实际属性: {[attr for attr in dir(call_result) if not attr.startswith('_')]}")
                        if hasattr(call_result, '__dict__'):
                            print(f"   实际数据: {call_result.__dict__}")
                        
                except Exception as e:
                    print(f"❌ 工具调用失败: {e}")
                    import traceback
                    traceback.print_exc()
                
                return True
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主函数"""
    print("=" * 60)
    print("MCP 客户端测试 (修复CallToolResult)")
    print("=" * 60)
    
    print("\n1. 测试基本API处理...")
    if not await test_with_proper_handling():
        print("\n基本API测试失败")
        return
    
    print("\n" + "="*60)
    print("🎉 测试完成！")
    print("="*60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n测试已停止")