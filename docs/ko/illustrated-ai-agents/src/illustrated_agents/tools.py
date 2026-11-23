import json
import inspect

from pathlib import Path
from typing import Any, Callable

from illustrated_agents.llm import Response


# 파이썬 자료형을 스키마의 자료형 이름으로 변환
TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool_to_schema(function: Callable) -> dict:
    """파이썬 함수를 OpenAI 호환 도구 스키마로 변환한다."""
    signature = inspect.signature(function)

    # 함수 매개변수 정보 추출
    properties, required = {}, []
    for name, parameter in signature.parameters.items():
        properties[name] = {"type": TYPE_MAP.get(parameter.annotation, "string")}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    # 스키마 구성
    schema = {
        "type": "function",
        "function": {
            "name": function.__name__,
            "description": inspect.getdoc(function),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

    return schema


class Tools:
    """에이전트가 사용할 도구를 등록하고 관리한다."""

    def __init__(self, requires_approval: list[str] = []):
        """도구 목록을 초기화하고 실행 전 승인이 필요한 도구를 지정한다."""
        self.registry = {}
        self.requires_approval = requires_approval

    def add_tool(self, name: str, func: Callable, description: str = "") -> None:
        """에이전트가 사용할 도구를 등록한다.

        매개변수:
            name: 도구의 이름.
            func: 도구의 동작을 구현한 함수.
            description: 도구의 용도.
        """
        self.registry[name] = {"function": func, "description": description}

    @property
    def descriptions(self) -> str:
        """등록한 모든 도구의 설명을 반환한다."""
        return "\n".join(
            f"`{tool}`: {self.registry[tool]['description']}"
            for tool in self.registry
        )

    @property
    def prompt(self) -> str:
        return f"""
# Tools

If needed, you can only use the following tools to assist you in completing tasks:

{self.descriptions}

To use a tool, respond with JSON: {{"tool": "name", "kwargs": {{"param": "value"}}}}
"""

    def parse(self, response: Response) -> Response:
        """텍스트에서 JSON 형식의 도구 호출을 추출한다."""
        text = response.content

        if '"tool":' in text or '"tool:"' in text:
            start, end = text.find("{"), text.rfind("}") + 1
            tool_call = json.loads(text[start:end])

            # 응답에 해석한 도구 호출 추가
            return Response(
                content=response.content,
                reasoning=response.reasoning,
                tool_call=tool_call,
            )

        return response

    def execute(self, response: Response) -> Any:
        """등록한 도구를 실행한다.

        매개변수:
            tool_call: "tool"과 "kwargs" 키를 담은 도구 호출 딕셔너리.
        """
        tool_call = response.tool_call
        name, kwargs = tool_call["tool"], tool_call.get("kwargs", {})

        # 승인이 필요한 도구는 실행 전에 사용자에게 확인
        if name in self.registry and name in self.requires_approval:
            response = input(f"{name} 실행을 허용할까요? [y/N] ").strip().lower()
            if response not in ("y", "yes"):
                return f"Tool '{name}' was denied by the user."

        # 등록한 도구 처리
        if name in self.registry:
            tool_func = self.registry[name]["function"]
            return tool_func(**kwargs)

        return f"Tool '{name}' not found."

    def observation(self, result: str) -> tuple[str, str]:
        """관찰 결과를 사용자 메시지로 반환한다."""
        return "user", f"OBSERVATION: {result}"

    def is_done(self, response: Response) -> bool:
        """TinyAgent가 실행을 마칠 조건을 확인한다."""
        if not response.tool_call:
            return True
        if response.tool_call["tool"] == "final_answer":
            response.content = response.tool_call.get("kwargs", "")
            return True
        return False

    @property
    def schemas(self) -> None:
        """모델 자체의 도구 호출에서만 사용한다."""
        return None


class NativeTools(Tools):
    """모델 자체의 함수 호출을 지원하는 도구 목록이다."""

    @property
    def schemas(self) -> list[dict]:
        """모델에 전달할 도구의 함수 스키마를 반환한다."""
        return [
            tool_to_schema(tool["function"]) for tool in self.registry.values()
        ]

    @property
    def prompt(self) -> str:
        """내장 도구 호출에는 별도 프롬프트가 필요하지 않아 빈 문자열을 반환한다."""
        return ""

    def parse(self, response: Response) -> Response:
        """도구 호출을 해석한다."""
        # 도구 호출이 없으면 받은 응답을 그대로 반환
        if not response.tool_call:
            return response

        # 도구 호출에서 이름과 인자 추출
        args = response.tool_call["function"]["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        tool_call = {
            "tool": response.tool_call["function"]["name"],
            "kwargs": args,
        }

        # 응답에 해석한 도구 호출 추가
        return Response(
            content=response.content,
            reasoning=response.reasoning,
            tool_call=tool_call,
        )

    def observation(self, result: str) -> tuple[str, str]:
        """내장 도구 호출의 결과는 'tool' 역할로 전달한다."""
        return "tool", str(result)

    def is_done(self, response: Response) -> bool:
        """도구 호출이 없으면 TinyAgent가 작업을 마친다."""
        return not response.tool_call


def _parse_frontmatter(text: str) -> dict:
    """외부 패키지 없이 YAML 머리말의 키와 값을 읽는다."""
    result = {}
    for line in text.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


class Skills(NativeTools):
    """도구와 스킬을 함께 관리한다.

    스킬은 특정 작업을 어떻게 수행할지 설명한다.
    스킬을 활성화하면 프로그램이 작업 지침을 반환한다.
    계산 도구와 스킬이 반환하는 값은 다르지만
    에이전트가 사용할 시점을 선택하는 방식은 같다.

    스킬의 내용은 필요한 시점에 읽는다.
    시스템 프롬프트에는 이름과 설명을 제공하고
    에이전트가 스킬을 호출하면 전체 지침을 전달한다.
    """

    def add_skill(self, path: str):
        """SKILL.md를 읽어 호출 가능한 도구로 등록한다."""
        content = Path(path).read_text(encoding="utf-8")

        # YAML 구분자를 기준으로 파일 머리말과 본문 지침 분리
        parts = content.split("---", 2)
        frontmatter = _parse_frontmatter(parts[1])
        name = frontmatter["name"]
        description = frontmatter["description"]
        instructions = parts[2].strip()

        # 스킬 등록
        def skill(**kwargs):
            return instructions

        skill.__name__ = name
        skill.__doc__ = f"A skill that when activated provides the following context: '{description}'"
        skill.__signature__ = (
            inspect.Signature()
        )  # 스키마에는 인자가 없지만 람다는 추가 인자를 받을 수 있음
        self.add_tool(name, skill, skill.__doc__)

    @property
    def prompt(self):
        return """You have specialized skills available. To use a skill,
call it like a tool by referencing their name.

The skill will provide detailed instructions for completing the task."""
