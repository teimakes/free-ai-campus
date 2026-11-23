import re
from illustrated_agents.llm import Response, LLM
from illustrated_agents.memory import Memory
from illustrated_agents.tools import Tools


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
