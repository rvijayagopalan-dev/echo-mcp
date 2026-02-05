import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def main():
    # Connect to the MCP server
    async with streamable_http_client("http://localhost:8000/mcp") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            init_result = await session.initialize()
            print(f"Initialized: {init_result}")

            # List available tools
            tools_result = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools_result.tools]}")

            # Call the echo tool
            call_result = await session.call_tool("echo", arguments={"message": "Hello from Python MCP client!"})
            print(f"Tool result: {call_result.content[0].text if call_result.content else 'No content'}")

if __name__ == "__main__":
    asyncio.run(main())