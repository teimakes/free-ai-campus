import urllib.request

from mcp.server.mcpserver import MCPServer


# MCP 서버 초기화
server = MCPServer("file_reader")


# 도구 정의
@server.tool()
def read_markdown(path: str) -> str:
    """Read the content of a markdown file.

    Args:
        path (str): The path to the markdown file.
    """
    with urllib.request.urlopen(path) as response:
        return response.read().decode()


# 서버 실행
def main():
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
