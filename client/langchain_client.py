import asyncio
import json
from typing import Any, Dict, List

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, Field

import dotenv
dotenv.load_dotenv()


class MCPToolWrapper(BaseTool):
    """LangChain tool that wraps an MCP tool."""

    name: str = Field(description="The name of the tool")
    description: str = Field(description="The description of the tool")
    mcp_tool: MCPTool = Field(description="The MCP tool object")
    session: ClientSession = Field(description="The MCP client session")

    def _run(self, **kwargs) -> str:
        """Run the MCP tool synchronously."""
        # Since MCP is async, we need to run it in an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._call_mcp_tool(kwargs))
            return result
        finally:
            loop.close()

    async def _call_mcp_tool(self, arguments: Dict[str, Any]) -> str:
        """Call the MCP tool asynchronously."""
        try:
            result = await self.session.call_tool(self.mcp_tool.name, arguments=arguments)
            if result.content:
                # Return the first content item's text
                for content in result.content:
                    if hasattr(content, 'text'):
                        return content.text
                    elif hasattr(content, 'type') and content.type == 'text':
                        return content.text
            return "Tool executed successfully"
        except Exception as e:
            return f"Error calling tool: {str(e)}"


async def create_langchain_tools_from_mcp(server_url: str) -> List[BaseTool]:
    """Create LangChain tools from MCP server tools."""
    tools = []

    async with streamable_http_client(server_url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize
            await session.initialize()

            # Get tools
            tools_result = await session.list_tools()

            # Create a persistent session for the tools
            # Note: This is a simplification; in production, you'd want better session management
            for mcp_tool in tools_result.tools:
                # Convert MCP tool input schema to LangChain format
                # For simplicity, we'll use a generic schema
                langchain_tool = MCPToolWrapper(
                    name=mcp_tool.name,
                    description=mcp_tool.description or f"Execute {mcp_tool.name} tool",
                    mcp_tool=mcp_tool,
                    session=session
                )
                tools.append(langchain_tool)

    return tools


async def main():
    # MCP server URL
    server_url = "http://localhost:8000/mcp"

    # Create LangChain tools from MCP
    tools = await create_langchain_tools_from_mcp(server_url)

    if not tools:
        print("No tools found from MCP server")
        return

    # Create LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to various tools."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Run the agent
    result = await agent_executor.ainvoke({
        "input": "Use the echo tool to say 'Hello from LangChain MCP client!'"
    })

    print("Final result:", result["output"])


if __name__ == "__main__":
    asyncio.run(main())