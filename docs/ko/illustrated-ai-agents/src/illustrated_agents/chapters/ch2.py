import json
import urllib.request

from dataclasses import dataclass

from illustrated_agents.utils import CodeAnnotator, DiffViewer, ChapterOverview
from illustrated_agents.chapters import ch1


@dataclass
class Response:
    """모델의 응답을 필드별로 저장한다."""

    content: str = ""
    reasoning: str | None = None
    tool_call: dict | None = None
    metadata: dict | None = None


class LLM:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "no_key",
        think: bool = False,
        temperature: float | None = None,
    ):
        """지정한 모델로 언어 모델 클라이언트를 초기화한다."""
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.think = think
        self.temperature = temperature

    def generate(
        self, messages: list[dict], tools: list | None = None
    ) -> Response:
        """메시지 목록을 모델에 보내 응답을 생성한다."""
        # 요청 본문 구성
        body = {
            "model": self.model,
            "messages": messages,
        }

        # 도구와 추론 설정
        if tools:
            body["tools"] = tools
        if not self.think:
            body["reasoning_effort"] = "none"
        else:
            body["reasoning_effort"] = "medium"
        if self.temperature is not None:
            body["temperature"] = self.temperature

        # OpenAI 호환 /chat/completions 엔드포인트로 POST 요청
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read())

        # 메시지, 도구 호출 및 응답 부가 정보 추출
        message = data["choices"][0]["message"]
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        tool_calls = message.get("tool_calls")
        tool_call = tool_calls[0] if tool_calls else None
        metadata = {
            "model": data["model"],
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"],
        }

        # Response 데이터 클래스에 결과 담기
        return Response(
            content=message.get("content"),
            reasoning=reasoning,
            tool_call=tool_call,
            metadata=metadata,
        )


@dataclass
class Step:
    """에이전트의 실행 기록에 들어가는 한 단계이다."""

    thought: str = ""
    action: dict | None = None
    observation: str | None = None
    answer: str | None = None
    metadata: dict | None = None


class Trajectory:
    """에이전트의 실행을 순서대로 기록한다."""

    def __init__(self) -> None:
        self.runs: list[dict] = []

    def initialize(self, query: str) -> None:
        """주어진 질문으로 새 실행 기록을 시작한다."""
        self.runs.append({"query": query, "steps": []})

    def add(self, response: Response, observation: str | None = None) -> None:
        """모델 응답을 한 단계로 기록하고, 관찰 결과가 있으면 함께 저장한다."""
        # THOUGHT 추가
        step = Step(
            thought=response.reasoning or "",
            metadata=response.metadata,
        )

        # ACTION과 OBSERVATION 또는 ANSWER 추가
        if observation is not None:
            step.action = response.tool_call
            step.observation = observation
        else:
            step.answer = response.content

        self.runs[-1]["steps"].append(step)


class TinyAgent:
    """교육용 에이전트의 기본 클래스를 정의한다."""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.memory = None  # 4장에서 메모리 추가
        self.tools = None  # 5장에서 도구 추가
        self.planner = None  # 6장에서 계획 기능 추가

        self.trajectory = Trajectory()

    def run(self, task: str) -> str:
        """주어진 작업으로 에이전트를 실행한다."""
        self.trajectory.initialize(task)
        return self._step(task)

    def _step(self, task: str) -> str:
        """에이전트의 한 단계를 실행한다."""
        messages = [{"role": "user", "content": task}]
        response = self.llm.generate(messages)
        self.trajectory.add(response)
        return response.content

    def _execute_action(self, action: str) -> str:
        """도구가 수행할 동작을 실행한다."""
        # 이후 장에서 구현할 부분
        return f"Executed action: {action}"

what_we_built = ChapterOverview(
    [
        ("agent.py", "updated", "`TinyAgent`에 `LLM`을 연결했습니다."),
        ("llm.py", "new", "로컬 또는 클라우드 모델을 호출하는 클래스입니다."),
        ("trajectory.py", "new", "에이전트가 수행한 단계를 순서대로 기록합니다."),
    ]
)

trajectory_add_annotated = CodeAnnotator(
    Trajectory.add,
    annotations={
        (
            4,
            7,
        ): "응답에 담긴 추론(THOUGHT)과 부가 정보를 단계 기록에 추가합니다.",
        (
            11,
            12,
        ): "도구 호출(ACTION)과 반환값(OBSERVATION)을 추가합니다.",
        14: "관찰 결과가 없으면 모델의 응답을 최종 답변(ANSWER)으로 기록합니다.",
        17: "완성한 단계를 목록 끝에 추가합니다.",
    },
)

llm_annotated = CodeAnnotator(
    LLM.generate,
    annotations={
        (
            12,
            19,
        ): "요청 본문에 도구와 추론 설정을 넣습니다. 표본 생성에 사용하는 temperature도 지정할 수 있습니다.",
        (
            22,
            31,
        ): "OpenAI 호환 API에 요청을 보내 응답을 받습니다.",
        (
            34,
            42,
        ): "응답에서 답변, 추론, 도구 호출 및 부가 정보를 추출합니다.",
        (
            45,
            50,
        ): "값을 `Response` 데이터 클래스에 담아 반환합니다.",
    },
)

tinyagents_diff = DiffViewer(
    ch1.TinyAgent, TinyAgent, "ch1.TinyAgent", "ch2.TinyAgent"
)
