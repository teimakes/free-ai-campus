from illustrated_agents.llm import EmbeddingModel, LLM


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
        super().add(role, content)

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


class LongTermMemory(Memory):
    """질문과 관련된 문서를 검색해 함께 전달하는 메모리이다."""

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

    def search(self, query: str, k: int = 3) -> list[str]:
        """질문과 유사도가 높은 문서 k개를 반환한다."""
        query_embedding = self.embedding_model.embed(query)
        scores = [self._cosine(query_embedding, emb) for emb in self.embeddings]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.documents[i] for i in ranked[:k]]

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
        return dot / norm


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


class MultiModalMemory(Memory):
    def add(
        self,
        role: str,
        content: str,
        tool_call: dict = None,
        image_data: str = None,
    ):
        """메모리에 메시지를 추가한다."""
        if image_data:
            is_url = image_data.startswith(("http://", "https://"))
            url = image_data if is_url else f"data:image/jpeg;base64,{image_data}"
            content = [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": content},
            ]
        super().add(role, content, tool_call=tool_call)
