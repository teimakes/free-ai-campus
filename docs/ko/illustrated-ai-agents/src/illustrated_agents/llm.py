import json
import urllib.request
from dataclasses import dataclass


@dataclass
class Response:
    """모델의 응답을 필드별로 저장한다."""

    content: str = ""
    reasoning: str | None = None
    tool_call: dict | None = None
    metadata: dict | None = None


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


class OpenAIClientLLM:
    def __init__(self, model: str, client, think: bool = False, **kwargs):
        """지정한 모델로 언어 모델 클라이언트를 초기화한다."""
        self.model = model
        self.client = client
        self.think = think
        self.kwargs = kwargs

    def generate(self, messages: list[dict], tools: list = None) -> Response:
        """메시지 목록을 모델에 보내 응답을 생성한다."""
        # 추론 활성화 여부 설정
        if self.think:
            extra_body = None
        else:
            extra_body = {
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_effort": "none",
            }

        # 응답 생성
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools if tools else None,
            extra_body=extra_body,
            **self.kwargs,
        )

        # 메시지, 도구 호출 및 응답 부가 정보 추출
        message = response.choices[0].message
        has_tool_call = hasattr(message, "tool_calls") and message.tool_calls
        tool_call = message.tool_calls[0].model_dump() if has_tool_call else None
        metadata = {
            "model": response.model,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

        # Response 데이터 클래스에 결과 담기
        return Response(
            content=message.content,
            reasoning=getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None),
            tool_call=tool_call,
            metadata=metadata,
        )


class EmbeddingModel:
    """임베딩을 생성한다."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
    ):
        """지정한 임베딩 모델로 클라이언트를 초기화한다."""
        self.model = model
        self.base_url = base_url

    def embed(self, text: str) -> list[float]:
        """텍스트를 수치 벡터로 변환한다."""
        # OpenAI 호환 /embeddings 엔드포인트로 POST 요청
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as resp:
            response = json.loads(resp.read())

        # 임베딩을 추출해 반환
        return response["data"][0]["embedding"]
