import argparse
import re


# Rich를 사용할 수 없으면 기본 콘솔 사용

try:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    Table = None

    class _PlainConsole:
        """서식 표시를 제거하고 출력하는 기본 콘솔이다."""

        _MARKUP = re.compile(r"\[/?[^\]]*\]")

        def print(self, *args, end="\n"):
            for arg in args:
                if isinstance(arg, str):
                    print(self._MARKUP.sub("", arg), end=end)
                # 문자열 이외의 Rule이나 Table 객체는 건너뜀

        @property
        def width(self):
            import shutil

            return shutil.get_terminal_size().columns

    console = _PlainConsole()


from illustrated_agents import LLM, TinyAgent, NativeReAct
from illustrated_agents.memory import MultimodalMemory
from illustrated_agents.display import (
    label,
    Display,
    RichDisplay,
    LOGO,
    FLAMINGO_LOGO,
)
from illustrated_agents.toolbox import create_native_code_tools


COMMANDS = {
    "/help": "사용 가능한 명령 표시",
    "/clear": "시스템 프롬프트를 보관하고 대화 기록 초기화",
    "/memory": "현재 대화 기록 표시",
    "/model": "현재 모델 이름 표시",
    "/tools": "사용 가능한 도구 목록",
    "/skills": "사용 가능한 스킬 목록",
}


def handle_command(cmd, agent):
    """슬래시 명령을 처리한다."""
    cmd = cmd.strip().lower()

    # /help로 사용 가능한 명령 조회
    if cmd == "/help":
        if HAS_RICH:
            table = Table(show_header=False, box=None, padding=(0, 2))
            for name, desc in COMMANDS.items():
                table.add_row(f"[bold]{name}[/]", f"[dim]{desc}[/]")
            console.print(table)
        else:
            for name, desc in COMMANDS.items():
                console.print(f"  {name:<12}  {desc}")

    # /clear로 시스템 메시지를 제외한 메모리 초기화
    elif cmd == "/clear":
        agent.memory.messages = agent.memory.messages[:1]
        console.print("  [dim]시스템 프롬프트를 보관하고 대화 기록을 초기화했습니다.[/]")

    # /memory로 현재 대화 기록 조회
    elif cmd == "/memory":
        messages = agent.memory.get_messages()
        console.print(f"  [dim]메모리에 저장된 메시지 {len(messages)}개[/]")
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = (
                msg["content"]
                if isinstance(msg["content"], str)
                else "(멀티모달 입력)"
            )
            style = (
                "blue"
                if role == "user"
                else "green"
                if role == "assistant"
                else "dim"
            )
            console.print(f"     [{style}]{i}: {role}: {content}[/]")

    # /model로 현재 모델 이름 조회
    elif cmd == "/model":
        console.print(f"  [dim]{agent.llm.model}[/]")

    # /tools로 도구 목록과 설명 조회
    elif cmd == "/tools":
        for name, tool in agent.tools.registry.items():
            console.print(f"  [yellow]{name}[/] - [dim]{tool['description']}[/]")

    # /skills로 스킬 목록과 설명 조회
    elif cmd == "/skills":
        for name, skill in agent.skills.skills.items():
            console.print(
                f"  [magenta]{name}[/] - [dim]{skill['description']}[/]"
            )

    # 알 수 없는 명령
    else:
        console.print(
            f"  [red]알 수 없는 명령 {cmd}[/] (명령 목록은 [bold]/help[/])"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--pink", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="언어 모델 이름",
        default="gemma4:e4b",
    )
    parser.add_argument(
        "-ab",
        "--api_base",
        type=str,
        help="API 기본 주소",
        default="http://localhost:11434/v1",
    )
    parser.add_argument(
        "-ak", "--api_key", type=str, help="API 키", default="no_key"
    )
    args = parser.parse_args()

    # Rich가 있으면 사용하고 그렇지 않으면 파이썬 ANSI 출력 사용
    display = (RichDisplay if HAS_RICH else Display)(pink=args.pink)

    # 언어 모델
    llm = LLM(
        model=args.model,
        base_url=args.api_base,
        api_key=args.api_key,
        think=True,
    )

    # 도구와 스킬
    tools = create_native_code_tools()

    # 모델의 추론과 도구 호출 필드를 사용하는 ReAct 플래너
    planner = NativeReAct(max_steps=10)

    # 코딩 에이전트
    agent = TinyAgent(
        llm=llm,
        memory=MultimodalMemory(),
        tools=tools,
        planner=planner,
        display=display,
    )

    # 모델, 도구, 스킬을 시작 화면에 표시
    tool_names = ", ".join(agent.tools.registry.keys())
    console.print(FLAMINGO_LOGO if args.pink else LOGO)
    console.print(f"\n  [dim]모델[/]  {args.model}")
    console.print(f"  [dim]도구[/]  {tool_names}")
    console.print("  [dim]명령 목록을 보려면[/] /help[dim]를 입력하세요.[/]\n")

    while True:
        # 사용자 입력
        try:
            w = console.width
            print(
                f"\033[48;5;237m \033[36m▎\033[97;1m >  {' ' * (w - 6)}\033[{w - 6}D",
                end="",
                flush=True,
            )
            query = input().strip()
            print("\033[0m")
        except (KeyboardInterrupt, EOFError):
            print("\033[0m", end="")
            console.print("  [dim]종료합니다.[/]")
            break

        # 빈 입력 건너뛰기
        if not query:
            continue

        # 종료 명령
        if query.lower() in ("exit", "quit"):
            console.print("  [dim]종료합니다.[/]")
            break

        # 슬래시 명령
        if query.startswith("/"):
            handle_command(query, agent)
            console.print()
            continue

        # 에이전트 실행
        try:
            result = agent.run(query)
            if hasattr(display, "_stop_thinking"):
                display._stop_thinking()
        except Exception as e:
            if hasattr(display, "_stop_thinking"):
                display._stop_thinking()
            if HAS_RICH:
                console.print(f"  {label('오류', 'red')}[red]{e}[/]\n")
            else:
                console.print(f"  오류    {e}\n")


if __name__ == "__main__":
    main()
