import re
from illustrated_agents.chapters import ch5
from illustrated_agents.chapters.ch2 import LLM, Response, Trajectory
from illustrated_agents.chapters.ch4 import Memory
from illustrated_agents.chapters.ch5 import Tools
from illustrated_agents.utils import DiffViewer, CodeAnnotator, ChapterOverview


class ReAct:
    """ReAct의 지침과 응답 해석 방식을 정의한다."""

    def __init__(self, max_steps: int = 10):
        """ReAct를 초기화한다.

        매개변수:
            max_steps: 한 작업에서 실행할 최대 단계 수.
        """
        self.max_steps = max_steps

    @property
    def prompt(self) -> str:
        return """
# ReAct (Reason and Act)

You are a ReAct agent that performs exactly ONE step per turn.

## ReAct Format

You use the following format for each step:

THOUGHT: [Your reasoning about what to do next]
ACTION:
{
    "tool": "a_tool_name",
    "kwargs": {"param": "value"},
}

An observation will be provided after each action. You do not generate the observation yourself.

## ReAct Completion

To provide the final answer to the task, use an action blob with "tool": "final_answer" tool. 
It is the only way to complete the task, else you will be stuck on a loop. 
So your final output should look like this:

ACTION:
{
    "tool": "final_answer",
    "kwargs": "insert your final answer here"
}

Use the `final_answer` tool when you are completely done with all subtasks and have the final answer ready.
You can also use `final_answer` to directly reply to a user's question without using any other tools.

"""

    def parse(self, response: Response) -> Response:
        """ReAct 응답을 THOUGHT와 ACTION으로 분리한다."""
        text = response.content

        # 각 구간을 찾는 패턴
        patterns = {
            "THOUGHT": r"THOUGHT:\s*(.+?)(?=ACTION:|OBSERVATION:|$)",
            "ACTION": r"ACTION:\s*(.+?)(?=THOUGHT:|OBSERVATION:|$)",
        }

        # 정규식으로 각 구간 추출
        result = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            result[key] = match.group(1).strip() if match else ""

        # 응답을 갱신하고 실행할 행동만 추출
        response.content = result["ACTION"]
        response.reasoning = result["THOUGHT"]
        return response


class NativeReAct(ReAct):
    """모델 자체의 추론 기능을 사용하는 ReAct이다."""

    @property
    def prompt(self) -> str:
        return ""

    def parse(self, response: Response) -> Response:
        return response


class PlanAndSolve(ReAct):
    """모델에 계획 작성 및 수정 도구를 제공하는 ReAct 플래너이다."""

    def __init__(self, tools: Tools, max_steps: int = 10):
        """PlanAndSolve 플래너를 초기화한다.

        매개변수:
            tools: update_plan과 mark_done을 추가할 도구 목록.
            max_steps: 한 작업에서 실행할 최대 단계 수.
        """
        super().__init__(max_steps)
        self.plan: list[dict] = []

        tools.add_tool(
            "update_plan",
            self.update_plan,
            "Create or overwrite the plan with a list of sub-tasks: update_plan(items: list[str])",
        )
        tools.add_tool(
            "mark_done",
            self.mark_done,
            "Mark a sub-task complete by 1-based index: mark_done(index: int)",
        )

    @property
    def prompt(self) -> str:
        return super().prompt + (
            "\n## Planning\n\n"
            "For non-trivial tasks, call `update_plan` first to outline your approach, "
            "then `mark_done` as you complete each sub-task. Revise the plan with "
            "`update_plan` whenever new information changes what is needed.\n"
        )

    def update_plan(self, items: list[str]) -> str:
        """Replace the plan with a new list of sub-tasks."""
        self.plan = [{"task": item, "done": False} for item in items]
        return f"Plan updated:\n{self._render()}"

    def mark_done(self, index: int) -> str:
        """Mark a sub-task complete by 1-based index."""
        self.plan[index - 1]["done"] = True
        return f"Plan:\n{self._render()}"

    def _render(self) -> str:
        return "\n".join(
            f"{'[x]' if item['done'] else '[ ]'} {i + 1}. {item['task']}"
            for i, item in enumerate(self.plan)
        )


class Reflexion:
    """최근 단계를 주기적으로 검토하고 수정 지침을 전달한다."""

    def __init__(self, llm: LLM, interval: int = 3):
        """Reflexion 평가기를 초기화한다.

        매개변수:
            llm: 검토 의견을 생성할 언어 모델.
            interval: 몇 단계마다 실행을 검토할지 지정하는 값.
        """
        self.llm = llm
        self.interval = interval

    def reflect(self, memory: Memory, step: int) -> None:
        """최근 단계를 검토해 수정 지침을 사용자 메시지로 메모리에 추가한다."""
        if step == 0 or step % self.interval:
            return
        critique = self.llm.generate(
            [
                {
                    "role": "system",
                    "content": "Critique the agent's recent steps and suggest a course correction.",
                },
                *memory.get_messages()[-6:],
            ]
        )
        memory.add("user", f"[Reflection] {critique.content}")


class TinyAgent:
    """교육용 에이전트의 기본 클래스를 정의한다."""

    def __init__(self, llm: LLM, memory: Memory, tools: Tools, planner: ReAct):
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = planner

        self.trajectory = Trajectory()

        # 시스템 프롬프트에 각 구성 요소의 지침을 추가
        system_prompt = "You are a helpful assistant.\n\n"
        system_prompt += self.planner.prompt
        system_prompt += self.tools.prompt
        self.memory.add("system", system_prompt)

    def run(self, task: str) -> str:
        """주어진 작업으로 에이전트를 실행한다."""
        self.memory.add("user", task)
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
        response = self.llm.generate(
            self.memory.get_messages(), tools=self.tools.schemas
        )
        self.memory.add(
            "assistant", response.content, tool_call=response.tool_call
        )

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
        result = self.tools.execute(response)

        # OBSERVATION: 도구 결과를 메모리에 추가하고 화면에 표시
        role, observation = self.tools.observation(result)
        self.memory.add(role, observation)
        self.trajectory.add(response, observation)

        return None


what_we_built_native = ChapterOverview(
    [
        ("agent.py", None, ""),
        ("llm.py", None, ""),
        ("memory.py", None, ""),
        ("planning.py", "new", "`NativeReAct` 클래스를 추가했습니다."),
        ("toolbox.py", None, ""),
        ("tools.py", None, ""),
        ("trajectory.py", None, ""),
    ]
)


what_we_built = ChapterOverview(
    [
        ("agent.py", "updated", "`for` 반복문과 ReAct 플래너로 여러 단계의 작업을 실행합니다."),
        ("llm.py", None, ""),
        ("memory.py", None, ""),
        ("planning.py", "new", "추론과 행동을 반복하는 ReAct를 추가했습니다."),
        ("toolbox.py", None, ""),
        ("tools.py", None, ""),
        ("trajectory.py", None, ""),
    ]
)


tinyagents_diff = DiffViewer(
    ch5.TinyAgent, TinyAgent, "ch5.TinyAgent", "ch6.TinyAgent"
)


react_prompt_annotated = CodeAnnotator(
    ReAct.prompt,
    annotations={
        (6, 7): "모델에 ReAct의 실행 절차를 설명합니다.",
        12: "<b>THOUGHT</b>에서 다음에 할 작업을 판단합니다.",
        (13, 18): "<b>ACTION</b>에 도구 호출 형식을 지정합니다.",
        19: "<b>OBSERVATION</b>은 도구 실행 후 프로그램이 전달합니다.",
        (
            26,
            30,
        ): "<b>ACTION</b>에서 최종 답변 형식을 선택해 작업을 끝냅니다.",
    },
)


react_parse_annotated = CodeAnnotator(
    ReAct.parse,
    annotations={
        (
            6,
            9,
        ): "응답에서 <b>THOUGHT</b>와 <b>ACTION</b>을 분리합니다. 모델이 임의로 생성한 관찰값을 쓰지 않도록 <b>OBSERVATION</b>에서 추출을 멈춥니다.",
        (14, 15): "정규식으로 각 구간을 찾습니다. re.DOTALL은 줄바꿈까지 일치시키고 re.IGNORECASE는 대소문자를 구분하지 않습니다. 일치하면 앞뒤 공백을 제거하고, 없으면 빈 문자열을 사용합니다.",
    },
)

tinyagent_init_annotated = CodeAnnotator(
    TinyAgent.__init__,
    annotations={
        (
            11,
            12,
        ): "시스템 메시지에 ReAct 지침을 추가하고 이어서 도구 사용 지침을 넣습니다.",
    },
)

tinyagent_run_annotated = CodeAnnotator(
    TinyAgent.run,
    annotations={
        3: "반복 중에 처음 작업을 참조하도록 사용자 메시지를 메모리에 먼저 저장합니다.",
        (
            7,
            8,
        ): "답변이 완성되거나 최대 단계 수에 도달할 때까지 ReAct 단계를 반복합니다.",
    },
)

tinyagent_step_annotated = CodeAnnotator(
    TinyAgent._step,
    annotations={
        (4, 5): "모델 응답을 생성해 메모리와 실행 기록에 반영합니다.",
        8: "ReAct 파서로 응답에서 THOUGHT와 ACTION을 추출합니다.",
        (
            12,
            14,
        ): "`final_answer`를 선택했으면 해당 단계를 기록하고 최종 답변을 반환해 반복을 종료합니다.",
        16: "`Response`에 도구 호출이 있으면 실행합니다.",
    },
)

tinyagent_action_annotated = CodeAnnotator(
    TinyAgent._execute_action,
    annotations={
        5: "`Tools`에서 도구를 이름으로 찾아 호출합니다.",
        (
            8,
            9,
        ): "실행 결과를 관찰 메시지로 저장하면 모델이 다음 단계에서 참조할 수 있습니다.",
        10: "해당 단계와 관찰 결과를 실행 기록에도 추가합니다.",
    },
)
