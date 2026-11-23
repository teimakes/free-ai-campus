from dataclasses import dataclass

from illustrated_agents.llm import Response


@dataclass
class Step:
    """에이전트의 실행 기록에 들어가는 한 단계이다."""

    thought: str = ""
    action: dict | None = None
    observation: str | None = None
    answer: str | None = None
    metadata: dict | None = None


class Trajectory:
    """에이전트의 실행을 순서대로 기록한다.

    각 실행은 "query"(str)와 "steps"(list[Step])를 담은 딕셔너리이다.
    """

    def __init__(self) -> None:
        self.runs: list[dict] = []

    def new_run(self, query: str) -> None:
        """주어진 질문으로 새 실행 기록을 시작한다."""
        self.runs.append({"query": query, "steps": []})

    def add_step(self, response: Response, observation: str | None = None) -> None:
        """응답에서 한 단계를 기록하며 도구 결과가 있으면 함께 저장한다."""
        step = Step(
            thought=response.reasoning or "",
            metadata=response.metadata,
        )
        if observation is not None:
            step.action = response.tool_call
            step.observation = observation
        else:
            step.answer = response.content
        self.runs[-1]["steps"].append(step)

    @property
    def steps(self) -> list[Step]:
        """모든 실행의 단계를 한 목록으로 반환한다."""
        return [step for run in self.runs for step in run["steps"]]

    @property
    def last_run(self) -> dict | None:
        """가장 최근 실행을 반환한다."""
        return self.runs[-1] if self.runs else None
