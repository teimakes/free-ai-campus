from dataclasses import dataclass
from typing import Callable

from illustrated_agents.utils import ChapterOverview

# 채점 함수의 타입 힌트
##  (prediction: str, example: dict) -> bool
Scorer = Callable[[str, dict], bool]


@dataclass
class Benchmark:
    name: str
    examples: list[dict]
    scorer: Callable


class Evaluator:
    """벤치마크의 각 문항을 TinyAgent로 실행하고 결과를 집계한다."""

    def __init__(self, create_agent: Callable):
        """새 에이전트를 만드는 함수를 전달받아 초기화한다."""
        self.create_agent = create_agent

    def run(self, benchmark: Benchmark) -> dict:
        """벤치마크의 각 예제를 실행하고 결과를 채점한다."""

        # 각 예제를 실행하고 결과 수집
        results = []
        for example in benchmark.examples:
            agent = self.create_agent()
            prediction = agent.run(example["task"]) or ""
            passed = benchmark.scorer(prediction, example)
            results.append(
                {
                    "prediction": prediction,
                    "passed": passed,
                }
            )

        # 통과율 계산
        if results:
            pass_rate = sum(result["passed"] for result in results) / len(results)
        else:
            pass_rate = 0.0

        # 상세 결과와 전체 통과율 반환
        return {
            "name": benchmark.name,
            "pass_rate": pass_rate,
            "results": results,
        }


what_we_built = ChapterOverview(
    [
        ("agent.py", None, ""),
        (
            "evaluator.py",
            "new",
            "`Benchmark`의 문제를 `TinyAgent`로 풀고 결과를 채점합니다.",
        ),
        ("llm.py", None, ""),
        ("memory.py", None, ""),
        ("planning.py", None, ""),
        ("toolbox.py", None, ""),
        ("tools.py", None, ""),
        ("trajectory.py", None, ""),
    ]
)
