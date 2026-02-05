from contextlib import asynccontextmanager
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

# Create MCP Server instance
mcp = FastMCP("vercel-mcp-server")

# Define a simple tool
@mcp.tool()
async def echo(message: str) -> str:
    """Echo back the provided message."""
    return f"Echo: {message}"

# Lifespan context manager for FastAPI
@asynccontextmanager
async def lifespan(app):
    async with mcp.session_manager.run():
        yield

# Create FastAPI app
app = FastAPI(lifespan=lifespan)

# Mount MCP server to FastAPI
app.mount("/", mcp.streamable_http_app())
