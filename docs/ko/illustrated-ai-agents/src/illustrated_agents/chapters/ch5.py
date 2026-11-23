import inspect
import json
from pathlib import Path
from typing import Any, Callable

from illustrated_agents.utils import DiffViewer, CodeAnnotator, ChapterOverview
from illustrated_agents.chapters import ch4
from illustrated_agents.chapters.ch2 import LLM, Response, Trajectory
from illustrated_agents.chapters.ch4 import Memory


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
            response = input(f"Allow {name}? [y/N] ").strip().lower()
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

    def add_skill(self, path: str) -> None:
        """SKILL.md를 읽어 호출 가능한 도구로 등록한다."""
        import yaml

        content = Path(path).read_text(encoding="utf-8")

        # YAML 구분자를 기준으로 파일 머리말과 본문 지침 분리
        parts = content.split("---", 2)
        frontmatter = yaml.safe_load(parts[1])
        name = frontmatter["name"]
        description = frontmatter["description"]
        instructions = parts[2].strip()

        # 스킬 등록
        def skill(**kwargs) -> str:
            return instructions

        skill.__name__ = name
        skill.__doc__ = f"A skill that when activated provides the following context: '{description}'"
        skill.__signature__ = (
            inspect.Signature()
        )  # 스키마에는 인자가 없지만 람다는 추가 인자를 받을 수 있음
        self.add_tool(name, skill, skill.__doc__)

    @property
    def prompt(self) -> str:
        return """You have specialized skills available. To use a skill,
call it like a tool by referencing their name.

The skill will provide detailed instructions for completing the task."""


class TinyAgent:
    """교육용 에이전트의 기본 클래스를 정의한다."""

    def __init__(self, llm: LLM, memory: Memory, tools: Tools):
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = None  # 6장에서 계획 기능 추가

        self.trajectory = Trajectory()

        # 시스템 프롬프트에 각 구성 요소의 지침을 추가
        system_prompt = "You are a helpful assistant.\n\n"
        system_prompt += self.tools.prompt
        self.memory.add("system", system_prompt)

    def run(self, task: str) -> str:
        """주어진 작업으로 에이전트를 실행한다."""
        self.memory.add("user", task)
        self.trajectory.initialize(task)

        return self._step()

    def _step(self) -> str:
        """에이전트의 한 단계를 실행한다."""
        # THOUGHT: 응답을 생성해 메모리에 추가
        response = self.llm.generate(
            self.memory.get_messages(), tools=self.tools.schemas
        )
        self.memory.add(
            "assistant", response.content, tool_call=response.tool_call
        )

        # 도구 호출 해석
        response = self.tools.parse(response)

        # ANSWER: 종료 조건 확인
        if self.tools.is_done(response):
            self.trajectory.add(response)
            return response.content

        return self._execute_action(response)

    def _execute_action(self, response: Response) -> str:
        """도구가 수행할 동작을 실행한다."""

        # ACTION: 도구 실행
        result = self.tools.execute(response)

        # OBSERVATION: 도구 결과를 메모리에 추가하고 화면에 표시
        role, observation = self.tools.observation(result)
        self.memory.add(role, observation)
        self.trajectory.add(response, observation)

        return observation


what_we_built = ChapterOverview(
    [
        ("agent.py", "updated", "`Tools`를 추가했습니다."),
        ("llm.py", None, ""),
        ("memory.py", None, ""),
        (
            "toolbox.py",
            "new",
            "에이전트가 호출할 도구 함수를 모아 둡니다.",
        ),
        (
            "tools.py",
            "new",
            "도구 등록, 응답 해석, 호출 실행을 담당하는 클래스를 만듭니다.",
        ),
        ("trajectory.py", None, ""),
    ]
)

what_we_built_native = ChapterOverview(
    [
        ("agent.py", None, ""),
        ("llm.py", None, ""),
        ("memory.py", None, ""),
        ("toolbox.py", None, ""),
        ("tools.py", "updated", "모델 자체의 도구 호출을 처리하는 `NativeTools`를 만듭니다."),
        ("trajectory.py", None, ""),
    ]
)

what_we_built_skills = ChapterOverview(
    [
        ("agent.py", None, ""),
        ("llm.py", None, ""),
        ("memory.py", None, ""),
        ("toolbox.py", None, ""),
        ("tools.py", "updated", "일반 도구와 스킬을 함께 관리하는 `Skills`를 추가합니다."),
        ("trajectory.py", None, ""),
    ]
)


tinyagents_diff = DiffViewer(
    ch4.TinyAgent, TinyAgent, "ch4.TinyAgent", "ch5.TinyAgent"
)


agent_init_annotated = CodeAnnotator(
    TinyAgent.__init__,
    annotations={
        (10, 12): "기본 지시와 도구 설명을 합쳐 시스템 프롬프트를 만듭니다. 초기화할 때 이를 메모리에 넣으면 모델이 첫 요청부터 도구를 참조할 수 있습니다.",
    },
)

agent_step_annotated = CodeAnnotator(
    TinyAgent._step,
    annotations={
        12: "응답에서 도구 이름과 인자를 읽습니다.",
        (15, 17): "모델의 응답에 도구 호출이 있으면 요청한 도구를 실행합니다.",
    },
)

agent_execute_action_annotated = CodeAnnotator(
    TinyAgent._execute_action,
    annotations={
        5: "(1) `Tools`가 등록한 도구를 찾아 호출합니다.",
        9: "(2) 도구가 반환한 값을 관찰 결과로 메모리에 저장합니다.",
        10: "(3) 도구 호출과 관찰 결과를 실행 기록에 추가합니다.",
    },
)


parse_tool_annotated = CodeAnnotator(
    Tools.parse,
    annotations={
        6: "간단한 JSON 문자열을 가정하고 중괄호로 둘러싸인 내용을 추출합니다.",
        (
            10,
            14,
        ): "해석한 도구 호출을 나중에 실행할 수 있도록 `Response`에 넣습니다.",
        16: "도구 호출이 없으면 원래 응답을 반환합니다.",
    },
)

execute_tool_annotated = CodeAnnotator(
    Tools.execute,
    annotations={
        (
            7,
            8,
        ): "도구 호출에서 이름과 인자를 꺼냅니다.",
        (
            11,
            14,
        ): "승인이 필요한 도구는 실행 전에 사용자에게 확인합니다. 10장에서 다룹니다.",
        (
            17,
            19,
        ): "등록한 도구를 이름으로 찾아 전달받은 인자로 실행합니다.",
        21: "해당 도구가 없으면 오류 메시지를 모델에 전달합니다.",
    },
)

observation_tool_annotated = CodeAnnotator(
    Tools.observation,
    annotations={
        3: "도구 결과를 사용자 메시지로 반환하면 모델이 다음 단계에서 그 값을 사용할 수 있습니다.",
    },
)

done_tool_annotated = CodeAnnotator(
    Tools.is_done,
    annotations={
        (3, 4): "도구 호출이 없으면 종료합니다.",
        (
            6,
            7,
        ): "6장에서는 `final_answer` 도구를 선택하면 최종 답변을 반환하고 종료합니다.",
    },
)
