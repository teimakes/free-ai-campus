from illustrated_agents.utils import CodeAnnotator, ChapterOverview


class TinyAgent:
    """교육용 에이전트의 기본 클래스를 정의한다."""

    def __init__(self):
        self.llm = None  # 2~3장에서 언어 모델 연결
        self.memory = None  # 4장에서 메모리 추가
        self.tools = None  # 5장에서 도구 추가
        self.planner = None  # 6장에서 계획 기능 추가

    def run(self, task: str) -> str:
        """주어진 작업으로 에이전트를 실행한다."""
        return self._step(task)

    def _step(self, task: str) -> str:
        """에이전트의 한 단계를 실행한다."""
        # 이후 장에서 구현할 부분
        return f"Received: {task}"

    def _execute_action(self, action: str) -> str:
        """도구가 수행할 동작을 실행한다."""
        # 이후 장에서 구현할 부분
        return f"Executed action: {action}"


what_we_built = ChapterOverview(
    [
        (
            "agent.py",
            "new",
            "`TinyAgent`의 기본 클래스와 메서드를 정의합니다.",
        ),
    ]
)


agent_annotated = CodeAnnotator(
    TinyAgent,
    annotations={
        (
            12,
            14,
        ): "`run`은 주어진 작업을 처리하기 위해 `_step`을 호출합니다. 이후 장에서는 한 작업에 여러 단계가 필요할 수 있습니다.",
        (
            16,
            19,
        ): "한 단계에서는 입력을 처리하고 응답을 생성합니다.",
        (21, 24): "`_execute_action`은 외부 API 호출이나 계산 같은 도구 실행을 위한 메서드입니다. 이후 장에서 구현합니다.",
    },
)
