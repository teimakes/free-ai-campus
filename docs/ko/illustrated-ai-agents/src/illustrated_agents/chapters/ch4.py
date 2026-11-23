import json
import urllib.request

from illustrated_agents.utils import CodeAnnotator, DiffViewer, ChapterOverview
from illustrated_agents.chapters.ch2 import LLM, Trajectory
from illustrated_agents.chapters import ch2


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


class Memory:
    """대화 이력을 저장하는 기본 메모리이다."""

    def __init__(self):
        self.messages = []

    def add(
        self, role: str, content: str, tool_call: dict | None = None, **kwargs
    ) -> None:
        """메모리에 메시지를 추가한다."""
        message = {"role": role, "content": content}

        # 도구 호출
        if tool_call:
            message["tool_calls"] = [tool_call]

        # 메모리에 메시지 추가
        self.messages.append(message)

    def get_messages(self) -> list[dict]:
        """저장된 메시지 전체를 반환한다."""
        return self.messages


class TrimmingMemory(Memory):
    """사용자와 모델의 최근 두 번의 문답을 보관한다."""

    def add(self, role: str, content: str, **kwargs) -> None:
        # 부모 클래스 Memory의 메서드로 새 메시지 추가
        super().add(role, content, **kwargs)

        # 시스템 메시지와 최근 두 번의 문답(메시지 네 개) 유지
        system = [
            message for message in self.messages if message["role"] == "system"
        ]
        turns = [
            message for message in self.messages if message["role"] != "system"
        ]
        self.messages = system + turns[-4:]


class SummarizationMemory(Memory):
    """언어 모델로 지난 대화를 요약하고 요약문을 갱신한다."""

    def __init__(self, llm: LLM):
        super().__init__()
        self.llm = llm

    def add(self, role: str, content: str, **kwargs) -> None:
        # 부모 클래스 Memory의 메서드로 새 메시지 추가
        super().add(role, content, **kwargs)

        # 문답이 끝날 때마다 누적 요약 갱신
        if role == "assistant":
            # 기존 요약과 새 대화 분리
            summary = ""
            conversation = ""
            for message in self.messages:
                if message["role"] == "system":
                    summary = message["content"]
                else:
                    conversation += f"{message['role']}: {message['content']}\n"

            # 기존 요약에 새 대화를 반영하도록 모델에 요청
            prompt = f"""Update the summary with the new conversation.

Summary: {summary}

Conversation:
{conversation}
Output the updated summary only."""
            response = self.llm.generate([{"role": "user", "content": prompt}])
            self.messages = [{"role": "system", "content": response.content}]


class RAGMemory(Memory):
    """검색 증강 생성으로 장기 메모리를 구현한다."""

    def __init__(self, embedding_model: EmbeddingModel, documents: list[str]):
        super().__init__()
        self.embedding_model = embedding_model
        self.documents = documents
        self.embeddings = [embedding_model.embed(doc) for doc in documents]

    def add(self, role: str, content: str, **kwargs) -> None:
        # 사용자 질문에 검색 결과를 추가한 뒤 저장
        if role == "user":
            context = "\n".join(self.search(content))
            content = f"""Context:
{context}

Question: {content}"""
        super().add(role, content, **kwargs)

    def search(self, query: str) -> list[str]:
        """질문과 유사도가 높은 문서 k개를 반환한다."""
        query_embed = self.embedding_model.embed(query)
        scores = [self._cosine(query_embed, embed) for embed in self.embeddings]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.documents[index] for index in ranked[:3]]

    def _cosine(self, a: list[float], b: list[float]) -> float:
        """두 임베딩 간의 코사인 유사도를 계산한다."""
        dot = sum(x * y for x, y in zip(a, b))
        norm = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
        return dot / norm


class TinyAgent:
    """교육용 에이전트의 기본 클래스를 정의한다."""

    def __init__(self, llm: LLM, memory: Memory):
        self.llm = llm
        self.memory = memory
        self.tools = None  # 5장에서 도구 추가
        self.planner = None  # 6장에서 계획 기능 추가

        self.trajectory = Trajectory()

    def run(self, task: str) -> str:
        """주어진 작업으로 에이전트를 실행한다."""
        self.memory.add("user", task)
        self.trajectory.initialize(task)

        return self._step()

    def _step(self) -> str:
        """에이전트의 한 단계를 실행한다."""
        # 응답을 생성해 메모리에 추가
        response = self.llm.generate(self.memory.get_messages())
        self.memory.add("assistant", response.content)
        self.trajectory.add(response)
        return response.content

    def _execute_action(self, action: str) -> str | None:
        """도구가 수행할 동작을 실행한다."""
        # 이후 장에서 구현할 부분
        return f"Executed action: {action}"


what_we_built = ChapterOverview(
    [
        ("agent.py", "updated", "`TinyAgent`에 `Memory`를 연결했습니다."),
        ("llm.py", None, ""),
        ("memory.py", "new", "대화 이력을 저장합니다."),
        ("trajectory.py", None, ""),
    ]
)

what_we_built_lt = ChapterOverview(
    [
        ("agent.py", None, ""),
        (
            "llm.py",
            "updated",
            "장기 메모리에서 사용할 `EmbeddingModel`을 추가합니다.",
        ),
        (
            "memory.py",
            "updated",
            "관련 문서를 검색하는 `RAGMemory`를 추가합니다.",
        ),
        ("trajectory.py", None, ""),
    ]
)


what_we_built_st = ChapterOverview(
    [
        ("agent.py", None, ""),
        ("llm.py", None, ""),
        (
            "memory.py",
            "updated",
            "`TrimmingMemory`와 `SummarizationMemory`를 추가합니다.",
        ),
        ("trajectory.py", None, ""),
    ]
)


tinyagent_annotated = CodeAnnotator(
    TinyAgent,
    annotations={
        14: "1. 사용자의 질문을 `Memory`에 추가합니다.",
        22: "2. 현재 메모리에 저장된 대화를 모델에 보내 응답을 생성합니다.",
        23: "3. 모델의 응답을 `Memory`에 추가합니다.",
    },
)


memory_annotated = CodeAnnotator(
    Memory,
    annotations={
        5: "메시지마다 발화자(role)와 내용(content)을 딕셔너리로 저장합니다.",
        11: "새 메시지의 발화자(예: user 또는 assistant)와 내용을 지정한 뒤 목록 끝에 추가합니다.",
        (
            14,
            16,
        ): "응답에 도구 호출이 있으면 함께 저장합니다. 5장에서 이 정보를 사용합니다.",
        (
            20,
            22,
        ): "대화 이력을 조회하면 저장한 메시지 목록을 반환합니다.",
    },
)

tinyagents_diff = DiffViewer(
    ch2.TinyAgent, TinyAgent, "ch2.TinyAgent", "ch4.TinyAgent"
)
