import subprocess
import sys
from pathlib import Path


from illustrated_agents.chapters import ch9
from illustrated_agents.chapters.ch2 import LLM, Response, Trajectory
from illustrated_agents.chapters.ch4 import Memory
from illustrated_agents.chapters.ch5 import Tools, NativeTools
from illustrated_agents.chapters.ch6 import ReAct, NativeReAct

from illustrated_agents.utils import CodeAnnotator, DiffViewer, ChapterOverview


BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[32m"  # THOUGHT: 생각
RED = "\033[31m"  # ACTION: 행동
YELLOW = "\033[33m"  # OBSERVATION: 관찰
PURPLE = "\033[35m"  # ANSWER: 답변


def read_file(path: str) -> str:
    """Read a file's contents."""
    target = Path(path)
    if not target.exists():
        return f"Error: '{path}' not found."
    return target.read_text(encoding="utf-8")


def list_files(directory: str = ".") -> str:
    """List files in a directory."""
    target = Path(directory)
    if not target.is_dir():
        return f"Error: '{directory}' is not a directory."
    entries = sorted(target.iterdir())
    return (
        "\n".join(p.name + ("/" if p.is_dir() else "") for p in entries)
        or "(empty)"
    )


def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Written to '{path}'."


def execute_python(code: str) -> str:
    """Execute Python code and return output."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (30s limit)."

    if result.returncode != 0:
        return (
            f"Exit code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        ).strip()
    return result.stdout.strip() or "(no output)"


class Display:
    """서식이 적용된 대화 화면이다."""

    def __call__(self, event: str, data: str | Response = None) -> None:

        # 응답 생성 중 상태 표시
        if event == "thinking":
            print(f"  Thinking...\n{RESET}")

        # THOUGHT: 생각
        elif event == "response":
            print(f"{BOLD}{GREEN}{'▒▒ THOUGHT ▒▒':<13}{RESET}")
            print(f"{data.reasoning}{RESET}\n")

            # ANSWER: 답변
            if data.content:
                print(f"{BOLD}{PURPLE}{'▒▒ ANSWER ▒▒':<13}{RESET}")
                print(f"{data.content}{RESET}\n")

        # ACTION: 행동
        elif event == "tool_call" and data:
            tool = data.tool_call["tool"]
            kwargs = data.tool_call["kwargs"]
            print(f"{BOLD}{RED}{'▒▒ ACTION ▒▒':<13}{RESET}")
            print(f"{tool}({kwargs}){RESET}\n")

        # OBSERVATION: 관찰
        elif event == "observation":
            print(f"{BOLD}{YELLOW}{'▒▒ OBSERVATION ▒▒':<13}{RESET}")
            print(f"{data}{RESET}\n")
            print(f"{'─' * 40}STEP{'─' * 40}{RESET}\n")


class TinyAgent:
    """교육용 에이전트의 기본 클래스를 정의한다."""

    def __init__(
        self,
        llm: LLM,
        memory: Memory,
        tools: Tools,
        planner: ReAct,
        display: Display,
    ):
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = planner
        self.display = display

        self.trajectory = Trajectory()

        # 시스템 프롬프트에 각 구성 요소의 지침을 추가
        system_prompt = "You are a helpful assistant.\n\n"
        system_prompt += self.planner.prompt
        system_prompt += self.tools.prompt
        self.memory.add("system", system_prompt)

    def run(self, task: str, image_data: str = None) -> str:
        """주어진 작업으로 에이전트를 실행한다."""
        self.memory.add("user", task, image_data=image_data)
        self.trajectory.initialize(task)

        # 여러 단계를 수행하는 반복문
        for step in range(self.planner.max_steps):
            result = self._step()
            if result is not None:
                return result

        return "Max steps reached without completion."

    def _step(self) -> str | None:
        """에이전트의 한 단계를 실행한다."""
        # THOUGHT: 응답을 생성해 메모리에 추가
        self.display("thinking")
        response = self.llm.generate(
            self.memory.get_messages(), tools=self.tools.schemas
        )
        self.memory.add(
            "assistant", response.content, tool_call=response.tool_call
        )
        self.display("response", response)

        # 도구 호출 해석
        response = self.planner.parse(response)
        response = self.tools.parse(response)

        # ANSWER: 종료 조건 확인
        if self.tools.is_done(response):
            self.trajectory.add(response)
            return response.content

        return self._execute_action(response)

    def _execute_action(self, response: Response) -> None:
        """도구가 수행할 동작을 실행한다."""

        # ACTION: 도구 실행
        self.display("tool_call", response)
        result = self.tools.execute(response)

        # OBSERVATION: 도구 결과를 메모리에 추가하고 화면에 표시
        role, observation = self.tools.observation(result)
        self.memory.add(role, observation)
        self.trajectory.add(response, observation)
        self.display("observation", observation)

        return None


NAME = """
██████ ██ ███  ██ ██  ██   ▄████▄  ▄████  ██████ ███  ██ ██████
  ██   ██ ██ ▀▄██  ▀██▀    ██▄▄██ ██  ▄▄▄ ██▄▄   ██ ▀▄██   ██
  ██   ██ ██   ██   ██     ██  ██  ▀███▀  ██▄▄▄▄ ██   ██   ██
"""  # 간단한 ANSI 출력


def main():
    import shutil

    # TinyAgent
    tools = NativeTools(requires_approval=["write_file", "execute_python"])
    tools.add_tool("read_file", read_file)
    tools.add_tool("list_files", list_files)
    tools.add_tool("write_file", write_file)
    tools.add_tool("execute_python", execute_python)
    agent = TinyAgent(
        llm=LLM(model="gemma4:e4b", think=True),
        memory=Memory(),
        tools=tools,
        planner=NativeReAct(),
        display=Display(),
    )

    print(f"\n{NAME}\n")

    while True:
        try:
            w = shutil.get_terminal_size().columns
            print(
                f"\033[48;5;237m \033[36m▎\033[97;1m >  {' ' * (w - 6)}\033[{w - 6}D",
                end="",
                flush=True,
            )
            query = input().strip()
            print("\033[0m")
        except (KeyboardInterrupt, EOFError):
            print("\033[0m", end="")
            break
        if not query or query.lower() in ("exit", "quit"):
            break
        try:
            agent.run(query)
        except Exception as e:
            print(f"ERROR: {e}\n")


what_we_built = ChapterOverview(
    [
        (
            "agent.py",
            "updated",
            "`TinyAgent`에 화면 출력을 연결합니다.",
        ),
        ("cli.py", "new", "터미널에서 출력 기능을 갖춘 `TinyAgent`를 실행합니다."),
        (
            "display.py",
            "new",
            "`Display`는 에이전트의 생각, 행동, 관찰 결과를 구분해 표시합니다.",
        ),
        ("llm.py", None, ""),
        ("memory.py", None, ""),
        ("planning.py", None, ""),
        ("skills.py", None, ""),
        ("toolbox.py", "updated", "코딩 에이전트가 사용할 도구를 추가했습니다."),
        ("tools.py", None, ""),
        ("trajectory.py", None, ""),
    ]
)


tinyagents_diff = DiffViewer(
    ch9.TinyAgent, TinyAgent, "ch9.TinyAgent", "ch10.TinyAgent"
)


display_annotated = CodeAnnotator(
    Display.__call__,
    annotations={
        5: "모델이 생성을 시작하면 처리 중이라는 상태를 표시합니다.",
        9: "응답이 도착하면 진행 표시를 멈추고 <b>THOUGHT</b>에 해당하는 내용을 출력합니다.",
        21: "중간 응답을 <b>ACTION</b>으로 표시하고 반복을 계속합니다.",
        26: "도구가 반환한 값을 <b>OBSERVATION</b>으로 표시합니다.",
    },
)

if __name__ == "__main__":
    main()
