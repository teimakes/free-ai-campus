from illustrated_agents.chapters import ch4, ch6
from illustrated_agents.chapters.ch2 import LLM, Response, Trajectory
from illustrated_agents.chapters.ch4 import Memory
from illustrated_agents.chapters.ch5 import Tools
from illustrated_agents.chapters.ch6 import ReAct
from illustrated_agents.utils import DiffViewer, ChapterOverview


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


class MultimodalMemory(Memory):
    """대화 이력을 저장하는 기본 메모리이다."""

    def add(
        self,
        role: str,
        content: str,
        tool_call: dict = None,
        image_data: str = None,
    ):
        """메모리에 메시지를 추가한다."""
        # 이미지
        if image_data:
            is_url = image_data.startswith(("http://", "https://"))
            url = image_data if is_url else f"data:image/jpeg;base64,{image_data}"
            content = [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": content},
            ]

        # 본문 메시지
        message = {"role": role, "content": content}

        # 도구 호출
        if tool_call:
            message["tool_calls"] = [tool_call]

        # 메모리에 메시지 추가
        self.messages.append(message)


tinyagents_diff = DiffViewer(
    ch6.TinyAgent, TinyAgent, "ch6.TinyAgent", "ch9.TinyAgent"
)
memory_diff = DiffViewer(
    ch4.Memory, MultimodalMemory, "ch4.Memory", "ch9.MultimodalMemory"
)


what_we_built = ChapterOverview(
    [
        (
            "agent.py",
            "updated",
            "텍스트와 함께 이미지를 처리하도록 에이전트를 수정합니다.",
        ),
        ("evaluator.py", None, ""),
        ("llm.py", None, ""),
        (
            "memory.py",
            "updated",
            "에이전트가 참조할 수 있도록 이미지를 대화 이력에 저장합니다.",
        ),
        ("planning.py", None, ""),
        ("toolbox.py", None, ""),
        ("tools.py", None, ""),
        ("trajectory.py", None, ""),
    ]
)
