from illustrated_agents.llm import LLM, Response, Trajectory
from illustrated_agents.memory import Memory
from illustrated_agents.planning import ReAct
from illustrated_agents.tools import Tools
from illustrated_agents.display import Display


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
