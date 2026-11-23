import sys
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from illustrated_agents.tools import Tools


class MCPTools(Tools):
    """Tools를 상속해 MCP 도구를 관리한다."""

    def __init__(self):
        super().__init__()
        self._load_mcp_tools()

    def _get_server_params(self):
        return StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "illustrated_agents.chapters.ch5_mcp_server",
            ],
        )

    def _load_mcp_tools(self):
        """MCP 서버에서 도구 목록을 가져와 등록한다."""

        # MCP 서버에서 도구 목록 조회
        async def fetch():
            async with stdio_client(self._get_server_params(), errlog=None) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    return tools.tools

        mcp_tools = asyncio.run(fetch())

        # 부모 클래스의 add_tool로 각 MCP 도구 등록
        for tool in mcp_tools:
            self.add_tool(
                name=tool.name,
                func=self._make_tool_caller(tool.name),
                description=tool.description,
            )

    def _make_tool_caller(self, name: str):
        """MCP 도구를 실행할 함수를 만든다."""

        def caller(**kwargs):
            return asyncio.run(self._call_tool(name, kwargs))

        return caller

    async def _call_tool(self, name: str, args: dict) -> str:
        """MCP 서버의 도구를 호출한다."""
        async with stdio_client(self._get_server_params(), errlog=None) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, args)
                return result.content[0].text
