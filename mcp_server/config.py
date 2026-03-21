"""MCP server configuration — extends base settings."""

from config.settings import Settings


class MCPSettings(Settings):
    mcp_base_url: str = ""  # Required for OAuth — must match public Railway URL
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8000


mcp_settings = MCPSettings()
