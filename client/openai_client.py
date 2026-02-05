import asyncio
import json
import os
from typing import Any, Dict, List

import openai
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Tool as MCPTool

import dotenv
dotenv.load_dotenv()


def convert_mcp_tool_to_openai_function(mcp_tool: MCPTool) -> Dict[str, Any]:
    """Convert MCP tool to OpenAI function format."""
    # Parse the inputSchema (assuming it's JSON Schema)
    try:
        schema = json.loads(mcp_tool.inputSchema) if isinstance(mcp_tool.inputSchema, str) else mcp_tool.inputSchema
    except:
        # Fallback schema if parsing fails
        schema = {
            "type": "object",
            "properties": {
                "arguments": {"type": "string", "description": "Arguments as JSON string"}
            },
            "required": ["arguments"]
        }

    return {
        "name": mcp_tool.name,
        "description": mcp_tool.description or f"Execute {mcp_tool.name} tool",
        "parameters": schema
    }


async def call_mcp_tool(session: ClientSession, tool_name: str, arguments: Dict[str, Any]) -> str:
    """Call an MCP tool and return the result."""
    try:
        result = await session.call_tool(tool_name, arguments=arguments)
        if result.content:
            # Return the first text content
            for content in result.content:
                if hasattr(content, 'text'):
                    return content.text
                elif hasattr(content, 'type') and content.type == 'text':
                    return content.text
        return "Tool executed successfully"
    except Exception as e:
        return f"Error calling tool: {str(e)}"


async def chat_with_mcp_tools():
    """Main chat loop with MCP tools integration."""
    # Set up OpenAI client
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # MCP server URL
    server_url = "http://localhost:8000/mcp"

    # Connect to MCP server
    async with streamable_http_client(server_url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize MCP connection
            await session.initialize()
            print("Connected to MCP server")

            # Get available tools
            tools_result = await session.list_tools()
            mcp_tools = tools_result.tools
            print(f"Available MCP tools: {[tool.name for tool in mcp_tools]}")

            # Convert to OpenAI functions
            functions = [convert_mcp_tool_to_openai_function(tool) for tool in mcp_tools]

            # Chat messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant with access to various tools. Use the available tools when appropriate."},
                {"role": "user", "content": "Use the echo tool to say 'Hello from OpenAI MCP client!'"}
            ]

            while True:
                # Get response from OpenAI
                response = await client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    functions=functions,
                    function_call="auto"
                )

                message = response.choices[0].message

                # Add assistant's message to conversation
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "function_call": message.function_call.model_dump() if message.function_call else None
                })

                # Check if function call is requested
                if message.function_call:
                    function_name = message.function_call.name
                    function_args = json.loads(message.function_call.arguments)

                    print(f"Calling MCP tool: {function_name} with args: {function_args}")

                    # Call the MCP tool
                    tool_result = await call_mcp_tool(session, function_name, function_args)

                    print(f"Tool result: {tool_result}")

                    # Add function result to messages
                    messages.append({
                        "role": "function",
                        "name": function_name,
                        "content": tool_result
                    })

                    # Continue the conversation with the tool result
                    continue
                else:
                    # No function call, print the response
                    print("Assistant:", message.content)
                    break


async def main():
    await chat_with_mcp_tools()


if __name__ == "__main__":
    asyncio.run(main())