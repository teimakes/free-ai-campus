import random
import time
import threading

from illustrated_agents.llm import Response


# 기본 Display에서 쓰는 ANSI 이스케이프 코드
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
RESET = "\033[0m"
ORANGE = "\033[38;5;208m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
PURPLE = "\033[35m"


# RichDisplay에서 사용하는 선택적 의존성
try:
    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    from rich.rule import Rule

    console = Console()
except ImportError:
    Console = Live = Text = Rule = console = None


NAME = """
██████ ██ ███  ██ ██  ██   ▄████▄  ▄████  ██████ ███  ██ ██████
  ██   ██ ██ ▀▄██  ▀██▀    ██▄▄██ ██  ▄▄▄ ██▄▄   ██ ▀▄██   ██
  ██   ██ ██   ██   ██     ██  ██  ▀███▀  ██▄▄▄▄ ██   ██   ██
"""  # 간단한 ANSI 출력
# 돌고래의 색상
_DK = "#3a5068"  # 어두운 색(등, 지느러미, 꼬리)
_DM = "#6a8498"  # 중간색(몸통과 머리)
_DL = "#a0b8c8"  # 밝은 색(배와 주둥이)

LOGO = f"""
[bold {_DK}]                     ▓▓[/]
[bold {_DK}]                       ▓▓[/][bold {_DM}]▒▒▒▒▒▒▒▒▒[/]
[bold {_DL}]                   ▒▒▒▒▒▒[/][bold {_DK}]▓[/][bold {_DM}]▒▒▒▒▒▒▒▒▒▒▒▒▒[/]
[bold {_DL}]                ▒▒▒▒▒▒▒▒▒▒[/][bold {_DK}]▓▓▓[/][bold {_DM}]▒▒▒▒▒▒▒▒▒▒▒▒▒[/]
[bold {_DL}]             ▒▒▒▒▒▒▒▒▒▒▒▒▒[/][bold {_DK}]▓▓██▓[/][bold {_DM}]▒▒▒▒▒▒▒▒▒▒▒▒▓[/]
[bold {_DL}]           ▒▒▒▒▒▒▒▒▒▒▒▒▒[/][bold {_DK}]▓███████▓[/][bold {_DM}]▒▒▒▓▓▓▓▓▓▒▒▓▒[/]
[bold {_DL}]          ▒▒▒▒▒▒▒▒▒[/][bold {_DK}]▓▓▓█████████▓[/][bold {_DM}]▒▒▒▒▒▒▒▒▒▓▓▒▒▓▓[/]
[bold {_DL}]         ▒▒▒▒▒▒[/][bold {_DK}]▓▓▓█████████████▓▓▓▓▓▓[/][bold {_DM}]▒▒▒▒▒▒▒▒▒▓▓[/]
[bold {_DL}]       ▒▒▒▒▒▒[/][bold {_DK}]▓███████████████████▓▓▓▓▓▓[/][bold {_DM}]▒▒▒▒▒▒▒▒▓[/]
[bold {_DL}]      ▒▒▒▒▒[/][bold {_DK}]▓███████████████████████▓▓▓▓▓[/][bold {_DM}]▒▒▒▒▒▒▒▓▓[/]
[bold {_DL}]     ▒▒▒▒[/][bold {_DK}]▓▓███████▓▓▓▓▓▓▓ █████▓[/][bold {_DM}]▒▒      ▓▓▓▒▒▒▒▒[/]
[bold {_DL}]     ▒▒▒[/][bold {_DK}]▓██████▓▓▓       ▓███▓[/][bold {_DM}]▒▒           ▓▓▒▒▒▒[/]
[bold {_DL}]     ▒▒[/][bold {_DK}]▓████▓▓▓        ▓██▓▓[/][bold {_DM}]▒▒                ▓▒[/]
[bold {_DL}]    ▒[/][bold {_DK}]▓▓████▓▓        ▓▓▓▓[/]
[bold {_DK}]    ▓▓████▓[/]
[bold {_DK}]    ▓████▓[/]
[bold {_DK}]   ▓▓███▓[/]
[bold {_DK}]▓██▓███▓[/]
[bold {_DK}]▓▓▓███▓[/]
[bold {_DK}]  ▓▓██▓▓[/]
[bold {_DK}]    ▓▓▓▓▒[/]
[bold {_DK}]      ▓▓▓▒[/]
[bold {_DK}]       ▓▓▒[/]
[bold {_DK}]        ▓▓[/]
[bold {_DK}]         ▓[/]
{NAME}"""  # https://www.asciiart.eu/image-to-ascii — 책 표지를 변환한 뒤 수작업으로 수정

# 홍학의 색상
_FH = "#f5c4bc"  # 머리
_FN = "#f0b0a8"  # 목
_FD = "#e8a098"  # 목 아랫부분
_FB = "#d4a090"  # 부리 시작 부분
_FK = "#424242"  # 부리 끝
_FE = "#000000"  # 눈

FLAMINGO_LOGO = f"""
      [bold {_FH}]            ░░░░░░░░░░[/]
      [bold {_FH}]         ▒▒░░░░░░░░░░░░░[/]
      [bold {_FH}]        ▒▒▒▒▒▒▒▒░▒▒░░░░░░[/]
      [bold {_FH}]      ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░[/]
      [bold {_FH}]      ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒[/][bold {_FE}]██[/][bold {_FH}]▒▒▒[/]
      [bold {_FN}]     ▒▒▒▒▒▒[/][bold {_FB}]▓▓▓▓[/][bold {_FN}]▒▒▒▒▒▒▒▒▒▒▒▒[/]
      [bold {_FN}]    ▒▒▒▒▒▒▒[/][bold {_FB}]▓▓▓▓▓▓[/][bold {_FN}]▒▒▒▒▒▒▒░░░[/]
      [bold {_FN}]    ▒▒▒▒▒▒▒▒   [/][bold {_FB}]▓▓▓▓[/][bold {_FN}]▒▒▒▒▒▒▒░░[/]
      [bold {_FN}]    ▒▒▒▒▒▒▒▒      [/][bold {_FK}]▓▓▓▓[/][bold {_FN}]▒▒▒▒▒▒░[/]
      [bold {_FD}]    ▒▒▒▒▒▒▒▒          [/][bold {_FK}]▓▒▒▓██▓▓[/]
      [bold {_FD}]    ▒▒▒▒▒▒▒░░          [/][bold {_FK}]███████[/]
      [bold {_FD}]    ▒▒▒▒▒▒▒░░░░        [/][bold {_FK}]██████[/]
      [bold {_FD}]    ▓▓▒▒▒▒▒▒▒░░░░      [/][bold {_FK}]██████[/]
      [bold {_FD}]      ▓▒▒▒▒▒▒▒▒░░░░░   [/][bold {_FK}]█████[/]
      [bold {_FD}]       ▓▓▒▒▒▒▒▒▒▒▒░░░░ [/][bold {_FK}]████[/]
      [bold {_FD}]         ▓▓▒▒▒▒▒▒▒▒▒▒▒▒[/][bold {_FK}]██[/]
      [bold {_FD}]           ▓▓▒▒▒▒▒▒▒▒▒▒▒[/]
      [bold {_FD}]             ▓▓▒▒▒▒▒▒▒▒▒▒▒[/]
      [bold {_FD}]              ▓▓▓▒▒▒▒▒▒▒▒▒[/]
      [bold {_FD}]               ▓▓▓▒▒▒▒▒▒▒▒▒[/]
      [bold {_FD}]                ▓▓▓▒▒▒▒▒▒▒▒[/]
      [bold {_FD}]                ▓▓▓▒▒▒▒▒▒▒▒▒[/]
      [bold {_FD}]                ▓▓▓▒▒▒▒▒▓▓▓▓[/]
{NAME}"""  # https://www.asciiart.eu/image-to-ascii — 10장의 홍학을 변환한 뒤 수작업으로 수정


def label(name: str, style: str) -> str:
    """단계 이름을 일정한 너비의 Rich 서식으로 표시한다."""
    return f"[bold {style}]{name:<13}[/]"


# 화면 출력 처리

THINKING_FRAMES = [
    "        ",
    " ~      ",
    " ~~     ",
    "~~~    ",
    "~~ ~~  ",
    "~  ~~ ~",
    "   ~~~ ",
    "     ~~",
    "        ",
]

THINKING_MESSAGES = [
    "Sonar pinging...",
    "Echolocating...",
    "Diving deep...",
    "Surfacing...",
    "Riding the current...",
    "Making a splash...",
    "Doing flips...",
    "Chasing fish...",
    "Blowing bubbles...",
    "Somersaulting...",
    "Fishing for answers...",
    "Navigating the depths...",
    "Scanning the ocean floor...",
    "Making waves...",
    "Deep dive...",
]

FLAMINGO_THINKING_MESSAGES = [
    "Standing on one leg...",
    "Preening feathers...",
    "Wading through data...",
    "Turning pink...",
    "Flocking together...",
    "Filter feeding...",
    "Stretching wings...",
    "Balancing act...",
    "Ruffling feathers...",
    "Doing the flamingo dance...",
    "Looking fabulous...",
    "Strutting around...",
    "Dipping beak...",
]


class Display:
    """ANSI 서식으로 에이전트의 실행 상태를 표시한다."""

    def __init__(self, pink=False):
        self._pink = pink

    def __call__(self, event: str, data: str | Response = None) -> None:

        # 응답 생성 중 상태 표시
        if event == "thinking":
            print(f"  {DIM}응답 생성 중...\n{RESET}")

        # THOUGHT 출력
        elif event == "response":
            print(f"{BOLD}{GREEN}{'▒▒ 추론 ▒▒':<13}{RESET}")
            print(f"{data.reasoning}{RESET}\n")
            if data.content:
                print(f"{BOLD}{PURPLE}{'▒▒ 답변 ▒▒':<13}{RESET}")
                print(f"{data.content}{RESET}\n")

        # ACTION 출력
        elif event == "tool_call" and data:
            tool = data.tool_call["tool"]
            kwargs = data.tool_call["kwargs"]
            print(f"{BOLD}{RED}{'▒▒ 행동 ▒▒':<13}{RESET}")
            print(f"{tool}({kwargs}){RESET}\n")

        # OBSERVATION 출력
        elif event == "observation":
            print(f"{BOLD}{YELLOW}{'▒▒ 관찰 결과 ▒▒':<13}{RESET}")
            print(f"{data}{RESET}\n")
            print(f"{DIM}{'─' * 80}{RESET}\n")


class RichDisplay:
    """Rich 서식으로 에이전트의 실행 상태를 표시한다. rich 패키지가 필요하다."""

    def __init__(self, pink=False):
        if Console is None:
            raise ImportError(
                "RichDisplay에는 rich 패키지가 필요합니다. "
                "설치 명령: pip install 'illustrated-agents[terminal]'"
            )
        self._live = None
        self._animating = False
        self._pink = pink

    def __call__(self, event: str, data: str | Response = None) -> None:

        # 응답 생성 중 애니메이션
        if event == "thinking":
            self._animating = True
            self._live = Live(
                console=console, refresh_per_second=8, transient=True
            )
            self._live.start()
            self._thinking_msg = random.choice(
                FLAMINGO_THINKING_MESSAGES if self._pink else THINKING_MESSAGES
            )
            threading.Thread(target=self._animate_thinking, daemon=True).start()

        # 텍스트 응답과 내장 추론 결과에서 THOUGHT 출력
        elif event == "response":
            self._stop_thinking()
            console.print(
                f"  [bold dark_orange]{'추론':<13}[/][dim italic]{data.reasoning}[/]"
            )
            console.print(
                f"\n  :flamingo: [bold magenta]답변[/]    {data.content}\n"
                if self._pink
                else f"\n  :dolphin: [bold cyan]답변[/]    {data.content}\n"
            )

        # ACTION 출력
        elif event == "tool_call" and data:
            tool = data.tool_call["tool"]
            kwargs = data.tool_call["kwargs"]
            console.print(
                f"  {label('행동', 'yellow')}[yellow]{tool}({kwargs})[/]"
            )

        # OBSERVATION 출력
        elif event == "observation":
            console.print(f"  {label('관찰 결과', 'green')}{data}\n")
            console.print(Rule(style="dim"), end="\n\n")

    def _animate_thinking(self) -> None:
        """응답을 기다리는 동안 물결 모양을 표시한다."""
        i = 0
        while self._animating and self._live:
            frame = THINKING_FRAMES[i % len(THINKING_FRAMES)]
            self._live.update(
                Text(f"  {self._thinking_msg}  {frame}", style="dark_orange")
            )
            time.sleep(0.15)
            i += 1

    def _stop_thinking(self) -> None:
        """응답 대기 표시를 멈춘다."""
        self._animating = False
        if self._live:
            self._live.stop()
            self._live = None
