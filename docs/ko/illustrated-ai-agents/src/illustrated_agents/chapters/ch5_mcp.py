from illustrated_agents.mcp import MCPTools
from illustrated_agents.utils import CodeAnnotator, ChapterOverview


what_we_built = ChapterOverview(
    [
        ("agent.py", None, ""),
        ("llm.py", None, ""),
        ("memory.py", None, ""),
        ("toolbox.py", None, ""),
        (
            "tools.py",
            "updated",
            "MCP 서버의 도구를 사용하기 위해 `MCPTools`를 추가했습니다.",
        ),
        ("trajectory.py", None, ""),
    ]
)


run_mcp_tool_annotated = CodeAnnotator(
    MCPTools,
    annotations={
        6: "서버에서 도구 목록을 불러오는 메서드입니다.",
        9: "실행할 MCP 서버의 위치를 지정합니다.",
        (
            15,
            20,
        ): "표준 메서드 `list_tools()`로 서버에서 도구 목록을 조회합니다. 각 항목에 도구 이름, 설명, 매개변수 명세가 들어 있습니다.",
        (
            25,
            30,
        ): "불러온 도구를 5장의 `Tools`에 등록합니다. `_make_tool_caller`가 이름과 인자를 서버로 전달할 함수를 만듭니다.",
        (
            42,
            46,
        ): "`session.call_tool`로 MCP 서버에 도구 이름과 인자를 전달하고 실행 결과를 받습니다.",
    },
)
