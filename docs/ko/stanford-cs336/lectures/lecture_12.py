from edtrace import text, link, image
from lecture_util import post_link
from references import mmlu_2021

def main():
    text("## 12강. 평가")
    text("- 지금까지 모델 구조, 학습, 시스템, 스케일링 등 언어 모델을 학습하는 방법을 다뤘다.")
    text("- 아직 다루지 않은 질문이 있다. 어떤 **데이터**로 학습할까?")
    text("- 데이터에 따라 모델의 행동이 달라진다. 코드, 다국어 텍스트, DNA 중 무엇을 사용할까?")
    text("- 데이터를 논하기 전에 모델이 어떻게 행동하기를 바라는지부터 살펴보자.")

    text("**평가**: 주어진 모델이 얼마나 **좋은가**?")

    what_is_good()

    perplexity()
    exam_benchmarks()
    chat_benchmarks()
    agentic_benchmarks()
    pure_reasoning_benchmarks()
    safety_benchmarks()

    realism()
    validity()
    how_to_think_about_evaluation()

    text("이 강의에서 살펴볼 점")
    text("- 모든 목적에 맞는 단 하나의 평가는 없다. 무엇을 측정할지에 따라 평가를 선택한다.")
    text("- 방법, 모델, 에이전트 중 무엇을 어떤 조건에서 비교하는지 분명히 한다.")
    text("- 난이도, 현실성, 타당도를 고려한다.")


def what_is_good():
    text("평가는 다음과 같은 기계적인 절차로 보일 수 있다.")
    text("1. 프롬프트를 정한다.")
    text("2. 모델에 보내 응답을 받는다.")
    text("3. 정확도를 계산한다.")

    text("하지만 평가는 더 깊이 다뤄야 할 중요한 주제다.")
    text("어떤 평가를 쓰느냐가 AI 개발 방향에 영향을 준다.")

    text("**평가의 과제**: <font color=\"red\">추상적인 개념</font> → <font color=\"blue\">구체적인 측정 지표</font>")

    text("벤치마크에서 높은 점수를 받으면 좋은 모델일까?")
    link(title="Artificial Analysis", url="https://artificialanalysis.ai/")
    image("images/artificial-analysis.png", width=800)

    text("벤치마크 점수가 높고 실행 비용이 낮으면 좋은 모델일까?")
    image("images/artificial-analysis-cost.png", width=800)

    text("사람들이 그 응답을 더 좋아하면 좋은 모델일까?")
    link(title="Arena AI(이전 이름 Chatbot Arena)", url="https://arena.ai/leaderboard")
    image("images/lmarena-leaderboard.png", width=400)

    text("사람들이 실제로 선택하고 비용을 지불하면 좋은 모델일까?")
    link(title="OpenRouter", url="https://openrouter.ai/rankings")
    image("images/openrouter.png", width=600)


def perplexity():
    text("- 언어 모델은 토큰 시퀀스에 대한 확률 분포 **p(x)**다.")
    text("- 퍼플렉서티 (1/p(D))^(1/|D|)는 모델 p가 데이터셋 D에 얼마나 높은 확률을 부여하는지 측정한다.")

    text("- 사전학습에서는 학습 세트의 퍼플렉서티를 최소화한다.")
    text("- 자연스럽게 생각할 수 있는 평가는 테스트 세트의 퍼플렉서티 측정이다.")
    text("- 전통적인 언어 모델링 연구에서 사용한 방법이다.")

    text("표준 데이터셋")
    text("- Penn Treebank(월스트리트 저널)")
    text("- WikiText-103(위키백과)")
    text("- One Billion Word Benchmark(기계 번역 WMT11의 EuroParl, 유엔, 뉴스 자료)")
    text("전통적인 방식은 동일 분포 평가다. 데이터셋의 학습 분할에서 학습한 뒤 같은 데이터셋의 테스트 분할에서 평가한다.")
    text("One Billion Word Benchmark에서 CNN과 LSTM만 사용해 퍼플렉서티를 51.3에서 30.0으로 낮췄다. "), link("https://arxiv.org/abs/1602.02410")

    text("GPT-2")
    text("- WebText로 학습했다. Reddit에서 링크한 웹사이트의 텍스트 40 GB다.")
    text("- 표준 데이터셋을 제로샷으로 평가했다(**분포 외** 평가).")
    image("images/gpt2-perplexity.png", width=800)
    text("- 전이 학습이 유리한 작은 데이터셋(PTB)에서는 성능이 좋아졌지만 큰 데이터셋(1BW)에서는 그렇지 않았다.")

    text("퍼플렉서티만 있으면 된다는 관점(과학적 주장보다 믿음에 가깝다)")
    text("- 실제 분포를 t, 모델을 p라 하자.")
    text("- 가능한 최선의 퍼플렉서티는 H(t)이며, p = t일 때만 얻는다.")
    text("- p = t라면 p(solution | problem)을 이용해 모든 과제를 풀 수 있다.")
    text("- 따라서 퍼플렉서티를 계속 낮추면 언젠가 ‘AGI에 도달할 것’이라는 주장이다.")

    text("퍼플렉서티가 필요 이상으로 많은 것을 평가할 수도 있다.")
    text("- 예문: *Stanford was founded in 1885*")
    text("- 퍼플렉서티는 모든 토큰의 예측을 평가한다. 예를 들어 *founded*의 예측은 관심 대상과 무관할 수 있다.")
    text("- 방법: 조건부 퍼플렉서티 p(response | prompt)^(1/|response|)를 측정한다.")

    text("일부 벤치마크는 실제로 퍼플렉서티를 다른 형식으로 측정한다.")
    text("- 빈칸 채우기: LAMBADA "), link("https://arxiv.org/abs/1606.06031")
    image("images/lambada.png", width=700)
    text("- 객관식 문장 완성: HellaSwag "), link("https://arxiv.org/pdf/1905.07830")
    image("images/hellaswag.png", width=500)

    text("**퍼플렉서티 순위표를 운영할 때 주의할 점**")
    text("- 참가자가 `LM`을 제출하면 운영자가 `log_prob = LM(test_data)`를 계산한다.")
    text("- 제출된 확률이 유효한 분포인지(합이 1인지) 신뢰할 수 있어야 한다.")
    text("- 하류 과제에서는 `response = LM(prompt)`로 응답을 받은 뒤 `response`의 정확도를 계산한다.")

    text("정리")
    text("- 퍼플렉서티는 스케일링 곡선이 매끄러워 언어 모델 개발에서 여전히 자주 쓴다.")
    text("- 퍼플렉서티만으로 충분하지 않다고 보면 실제 사용 상황을 반영하는 벤치마크도 필요하다.")


def exam_benchmarks():
    text("사람을 평가할 때처럼 시험으로 언어 모델을 평가할 수 있다.")
    text("- 과목과 난이도를 조절할 수 있다.")
    text("- 정답이 명확하고 채점하기 쉽게 설계할 수 있다.")

    text("**Massive Multitask Language Understanding (MMLU)** "), link(mmlu_2021)
    text("- 수학, 미국사, 법학, 도덕 등 57개 과목의 객관식 문제다.")
    text("- 학부생과 대학원생이 인터넷에서 자유롭게 이용할 수 있는 자료에서 문제를 수집했다.")
    text("- 이름에는 언어 이해가 들어가지만 MMLU가 주로 측정하는 것은 지식이다.")
    text("- GPT-3를 퓨샷 프롬프팅으로 평가했다.")
    image("images/mmlu.png", width=700)
    link("https://llm-stats.com/benchmarks/mmlu")
    link(title="HELM에서 MMLU 예측 결과 보기", url="https://crfm.stanford.edu/helm/mmlu/latest/")

    text("**MMLU-Pro** "), link("https://arxiv.org/abs/2406.01574")
    text("- MMLU에서 오류가 있거나 지나치게 쉬운 문제를 제외했다.")
    text("- 선택지를 4개에서 10개로 늘렸다.")
    text("- 모델에 풀이 기회를 더 주기 위해 사고 과정(chain of thought)을 사용해 평가했다.")
    text("- 모델 정확도가 16~33% 하락해 점수가 상한에 몰리는 현상이 줄었다.")
    image("images/mmlu-pro.png", width=700)
    link("https://llm-stats.com/benchmarks/mmlu-pro")
    link(title="HELM에서 MMLU-Pro 예측 결과 보기", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/mmlu_pro")

    text("**Graduate-Level Google-Proof Q&A (GPQA)** "), link("https://arxiv.org/abs/2311.12022")
    text("- Upwork에서 모집한 박사급 전문가 61명이 문제를 작성했다.")
    image("images/gpqa.png", width=700)
    text("- 해당 분야 박사 전문가의 정확도는 65%다.")
    text("- 비전문가는 Google 검색을 사용할 수 있는 30분 동안 34%의 정확도를 기록했다.")
    text("- GPT-4의 정확도는 39%다.")
    link("https://llm-stats.com/benchmarks/gpqa")
    link(title="HELM에서 GPQA 예측 결과 보기", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/gpqa")

    text("**Humanity's Last Exam (HLE)** "), link("https://arxiv.org/abs/2501.14249")
    text("- 여러 과목에 걸친 멀티모달 객관식·단답형 문제 2,500개다.")
    image("images/hle-examples.png", width=700)
    text("- 출제자에게 총상금 50만 달러와 공동 저자 자격을 제공했다.")
    text("- 최첨단 언어 모델로 문제를 거르고 여러 단계로 검토했다.")
    image("images/hle-pipeline.png", width=700)
    image("images/hle-results.png", width=600)
    link("https://llm-stats.com/benchmarks/hle")

    text("정리")
    text("- 모델 성능이 향상되어 기존 벤치마크의 상한에 도달하면 더 어려운 문제를 사용한다.")
    text("- 객관식으로도 원하는 만큼 어려운 문제를 만들 수 있다.")
    text("- 다만 실제 사용에서는 열린 질문이 많고 하나의 정답이 존재하지 않을 수 있다.")


def chat_benchmarks():
    text("- 지금까지는 정답이 명확한 객관식 과제를 평가했다.")
    text("- 대부분의 사용자가 AI 비서에 객관식 시험 문제를 묻는 것은 아니다.")
    
    text("예")
    text("프롬프트: *I would like to make a beet salad with goat cheese. What kind of herbs would work well and what would not work well?*")
    text("응답: *Here’s a breakdown of herbs that work well (and some that don’t) in a beet + goat cheese salad, based on how their flavors interact with the sweet-earthiness of beets and the tangy creaminess of goat cheese...")

    text("**질문**: 자유 형식 응답을 어떻게 평가할까?")

    text("**Chatbot Arena** "), link("https://arxiv.org/abs/2403.04132")
    text("데이터 수집")
    text("- 인터넷 사용자가 프롬프트를 직접 입력한다.")
    text("- 무작위로 고른 익명의 모델 두 개가 답한다.")
    text("- 사용자가 어느 응답이 더 나은지 평가한다.")
    image("images/arena-beets.png", width=700)
    text("쌍별 비교를 이용해 Elo 순위를 계산한다.")
    text("- 모델을 정의한다: p(A가 B를 이김) = 1 / (1 + 10^((ELO_B - ELO_A)/400)).")
    text("- 관찰된 쌍별 비교 결과의 확률을 최대화하도록 적합한다.")
    link(title="Arena AI(이전 이름 Chatbot Arena)", url="https://arena.ai/leaderboard")
    image("images/lmarena-leaderboard.png", width=400)
    text("특성")
    text("- 실제 사용자가 입력한 프롬프트다. 무료로 제공하므로 실제로 사용할 유인이 있다.")
    text("- 어떤 사용자가 참여하는가? 편향이나 스팸 참여자는 없는가?")
    text("- 둘 중 하나를 고르게 하면 문체와 내용의 정확성을 구별하기 어렵다.")
    text("- 사용자가 정답 여부를 판단할 수 있을까? 사용자에게 동조하는 응답이 유리할까?")
    text("- 모든 모델에 같은 프롬프트를 줄 필요가 없다. 사람이 직접 평가하므로 중요한 특성이다.")
    text("- 시간이 지나면서 새 프롬프트와 모델이 계속 추가된다.")

    text("**AlpacaEval** (2023)"), link(title="순위표", url="https://tatsu-lab.github.io/alpaca_eval/")
    text("- 여러 출처에서 수집한 지시문 805개다.")
    text("- GPT-4 preview가 판정한 기준 모델(GPT-4 preview) 대비 승률을 사용한다. 편향이 생길 수 있을까?")
    text("- 언어 모델 판정자는 긴 응답을 선호한다. 이를 이용해 순위를 높이는 경우가 있었다.")
    text("- AlpacaEval 2.0은 회귀로 이 편향을 보정했다. "), link("https://arxiv.org/pdf/2404.04475")
    text("- 평가 지표 자체는 어떻게 평가할까?")
    text("- 사람의 선호를 반영하는 Chatbot Arena 결과와 상관관계가 높다.")
    image("https://github.com/tatsu-lab/alpaca_eval/raw/main/figures/chat_correlations_no_ae.png", width=500)
    image("images/alpacaeval-leaderboard.png", width=400)

    text("**WildBench** "), link("https://arxiv.org/pdf/2406.04770")
    text("- 사람과 챗봇 사이의 대화 100만 건에서 예제 1,024개를 추렸다.")
    text("- 점검 목록을 사용하는 GPT-4 Turbo(판정용 사고 과정과 비슷함)와 GPT-4를 판정자로 쓴다.")
    text("- Chatbot Arena와 상관관계가 높다. 이 비교가 사실상 결과의 타당성을 확인하는 방법이 된 듯하다.")
    image("images/wildbench.png", width=700)
    link(title="HELM에서 WildBench 예측 결과 보기", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/wildbench")

    text("정리")
    text("- 자유 형식 응답을 어떻게 평가할까?")
    text("- 비슷한 응답을 쌍으로 비교하면 차이를 더 잘 포착할 수 있다.")
    text("- 사람 평가자와 언어 모델 평가자 모두 편향이 있을 수 있다.")
    text("- 사람이든 모델이든 점검 목록과 채점 기준을 사용하면 신뢰도가 높아진다.")


def agentic_benchmarks():
    text("앞에서는 언어 모델이 하는 말(채팅)을 평가했다.")
    text("이제는 언어 모델이 하는 일(에이전트)을 평가한다.")

    text("에이전트 = 언어 모델 + 에이전트 실행 구조(모델을 어떻게 사용할지 정하는 로직).")
    
    text("코드 실행 같은 도구 사용과 여러 차례의 작업이 필요한 과제를 생각해 보자.")

    text("**SWEBench** "), link("https://arxiv.org/abs/2310.06770")
    text("- Python 저장소 12개의 과제 2,294개다.")
    text("- 코드베이스와 이슈 설명을 받고 PR을 제출한다.")
    text("- 단위 테스트로 평가한다.")
    image("images/swebench.png", width=800)
    link("https://llm-stats.com/benchmarks/swe-bench-verified")

    text("**TerminalBench** "), link("https://arxiv.org/abs/2601.11868"), link(title="웹사이트", url="https://www.tbench.ai/")
    image("images/terminal-bench.png", width=700)
    text("- 컴퓨터 터미널은 단순하고 범용적인 환경이다.")
    text("- 기여자 93명이 과제 229개를 모았고 그중 89개로 Terminal-Bench 2.0을 구성했다.")
    image("images/terminal-bench-human-time.png", width=600)
    image("images/terminal-bench-results.png", width=600)
    link("https://llm-stats.com/benchmarks/terminal-bench")

    text("**CyBench** "), link("https://arxiv.org/abs/2408.08926")
    image("images/cybench.png", width=700)
    text("- CTF(Capture the Flag) 과제 40개다.")
    text("- 최초 풀이까지 걸린 시간으로 난이도를 측정한다.")
    image("images/cybench-agent.png", width=700)
    image("images/cybench-results.png", width=600)
    link("https://llm-stats.com/benchmarks/cybench")

    text("**MLEBench** "), link("https://arxiv.org/abs/2410.07095")
    text("- 모델 학습과 데이터 처리 등이 필요한 Kaggle 대회 75개다.")
    image("images/mlebench.png", width=800)
    image("images/mlebench-results.png", width=700)

    text("에이전트의 실행 구조 "), post_link("https://www.philschmid.de/agents-2.0-deep-agents")
    image("https://www.philschmid.de/static/blog/agents-2.0-deep-agents/overview.png", width=400)
    text("- 명시적 계획: 할 일 목록을 두고 완료한 항목을 표시한다.")
    text("- 계층적 위임: 에이전트가 별도 문맥을 갖는 하위 에이전트를 호출한다.")
    text("- 지속 메모리: 파일을 읽고 쓴다.")
    text("- 세밀한 문맥 설계: 작업 절차에 대한 지시를 자세히 제공한다.")

    text("정리")
    text("- 에이전트는 언어 모델로 수행할 수 있는 작업의 범위를 크게 넓힌다.")
    text("- 에이전트를 실행하는 구조가 결과에 큰 영향을 준다.")
    text("- 에이전트 평가는 실행 구조와 언어 모델을 함께 평가한다.")


def pure_reasoning_benchmarks():
    text("- 지금까지의 과제는 모두 언어 지식과 세상에 관한 지식을 필요로 했다.")
    text("- **추론**을 지식과 분리해 평가할 수 있을까?")
    text("- 추론은 사실을 외우는 능력보다 지능 자체를 더 직접적으로 보여 준다고 볼 수 있다.")

    text("**ARC-AGI** "), link(title="웹사이트", url="https://arcprize.org/arc-agi")
    text("- 사람은 100\\% 풀 수 있으나 AI에는 어렵다.")
    text("- 모든 과제가 서로 달라 암기가 도움이 되지 않는다.")

    text("- ARC-AGI-1(2019): 첫 버전.")
    image("https://arcprize.org/media/images/arc-task-grids.jpg", width=800)

    text("- ARC-AGI-2(2025년 3월): 여러 단계의 추론을 더 많이 요구한다.")
    image("https://arcprize.org/media/images/blog/arc-agi-2-unsolved-1.png", width=800)

    image("images/arc-agi-results.png", width=700)
    text("- 사전학습 언어 모델만으로는 뚜렷한 개선이 없었다.")
    text("- o1, o3 같은 추론 모델부터 성능이 크게 향상되기 시작했다.")

    text("- ARC-AGI-3(2026년 3월): 상호작용 환경. "), post_link("https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf")
    image("images/arc-agi-3.png", width=300)
    image("images/arc-agi-3-results.png", width=500)

    text("정리")
    text("- 목표는 지식과 추론을 분리하는 것이지만 쉽지 않다.")
    text("- 인간의 추론에 맞춘 과제이므로 초인적 추론까지 측정하지는 않는다.")
    text("- 현재 모델이 부족한 부분을 분명히 보여 준다.")


def safety_benchmarks():
    image("https://www.team-bhp.com/forum/attachments/road-safety/2173645d1625144681-will-crash-test-rating-change-if-higher-variant-chosen-images-30.jpeg", width=400)
    text("AI에서 안전성이란 무엇일까?")

    text("**HarmBench** "), link("https://arxiv.org/abs/2402.04249")
    text("- 법이나 규범을 위반하는 유해 행동 510가지를 바탕으로 구성했다.")
    link(title="HELM에서 HarmBench 보기", url="https://crfm.stanford.edu/helm/safety/latest/#/leaderboard/harm_bench")
    link(title="안전성 실패의 예", url="https://crfm.stanford.edu/helm/safety/latest/#/runs/harm_bench:model=anthropic_claude-3-7-sonnet-20250219?instancesPage=4")

    text("**AIR-Bench** "), link("https://arxiv.org/abs/2407.17436")
    text("- 규제 체계와 기업 정책을 바탕으로 만들었다.")
    text("- 위험 유형 314개로 분류한 프롬프트 5,694개다.")
    image("https://crfm.stanford.edu/helm/assets/air-overview-DpBbyagA.png", width=800)
    link(title="HELM에서 AIR-Bench 보기", url="https://crfm.stanford.edu/helm/air-bench/latest/#/leaderboard")

    text("탈옥(jailbreaking)")
    text("- 언어 모델은 유해한 지시를 거부하도록 학습한다.")
    text("- GCG(Greedy Coordinate Gradient)는 안전 장치를 우회하는 프롬프트를 자동으로 최적화한다. "), link("https://arxiv.org/pdf/2307.15043")
    text("- 가중치 공개 모델(Llama)에서 나타난 효과가 비공개 모델(GPT-4)로 전이된다.")
    image("images/gcg-examples.png", width=800)

    text("안전성이란 무엇인가?")
    text("- 정치, 법률, 사회 규범처럼 맥락에 크게 의존하며 국가별로 달라지는 측면이 많다.")
    text("- 위험도 환각, 아첨, 범죄 방조, 불평등, 비판적 사고력 저하 등 다양하다.")

    text("**이중 용도**: 성능이 높은 사이버 보안 에이전트(Mythos)는 시스템 침입에도, 모의 침투 점검에도 쓰일 수 있다.")


def realism():
    text("**생태 타당도**: 평가가 실제 사용 상황을 얼마나 잘 반영하는가?")
    text("- GPQA 같은 시험 벤치마크는 실제 사용과 거리가 있다.")
    text("- Chatbot Arena 프롬프트는 실제 사람이 작성하지만 분포를 통제하지 않는다.")

    text("**GDPVal** (OpenAI) "), link("https://arxiv.org/pdf/2510.04374")
    text("- 미국 GDP 상위 9개 부문의 직종 44개를 다룬다.")
    text("- 경력이 약 14년인 전문가들이 과제를 제공했다.")
    image("images/gdpval.png", width=700)

    text("**MedHELM** "), link("https://arxiv.org/abs/2505.23802")
    text("- 기존 의료 벤치마크는 표준화 시험을 바탕으로 했다.")
    text("- 임상의 29명이 제시한 임상 과제 121개다. 비공개·공개 데이터셋을 함께 사용한다.")
    image("https://crfm.stanford.edu/helm/assets/medhelm-overview-CND0EIsy.png", width=700)
    link(title="MedHELM", url="https://crfm.stanford.edu/helm/medhelm/latest/#/leaderboard")

    text("**Clio** (Anthropic) "), link("https://arxiv.org/abs/2412.13678")
    text("- 언어 모델로 실제 사용자 데이터를 분석한다.")
    text("- 사람들이 어떤 요청을 하는지 전반적인 양상을 공개한다.")
    image("images/clio-table4.png", width=700)

    text("현실성과 개인정보 보호가 서로 충돌할 때도 있다.")


def validity():
    text("평가의 타당성을 어떻게 알 수 있을까?")

    text("### 학습 데이터와 테스트 데이터의 중복")
    text("- 기계학습의 기초: 테스트 세트로 학습하지 않는다.")
    text("- 기반 모델 이전에는 ImageNet이나 SQuAD처럼 학습·테스트 분할이 명확했다.")
    text("- 오늘날에는 인터넷 데이터로 학습하고 어떤 데이터를 썼는지 공개하지 않는 경우가 있다.")

    text("방법 1. 모델을 보고 학습·테스트 중복을 추정한다.")
    text("- 데이터 점들의 교환 가능성을 이용한다. "), link("https://arxiv.org/pdf/2310.17623")
    image("images/contamination-exchangeability.png", width=500)

    text("방법 2. 신뢰 구간을 보고하듯 중복도 보고하는 관행을 만든다.")
    text("- 모델 제공자가 학습·테스트 데이터의 중복을 보고한다. "), link("https://arxiv.org/abs/2410.08385")

    text("방법 3. 최신 평가 자료를 사용한다.")
    text("- LiveCodeBench와 UncheatableEval은 새로 게시된 웹페이지에서 자료를 수집한다.")
    text("- 이전 자료를 복사한 페이지도 있어 게시 시각만으로는 보장할 수 없다.")

    text("방법 4. 비공개 평가 자료를 사용한다.")
    text("- 기업에서는 인터넷에 없는 내부 코드베이스를 사용한다.")
    text("- 개인적으로 작성한 글을 사용할 수 있다.")
    text("- 퍼플렉서티 평가에서 가장 적용하기 쉽다.")

    text("### 데이터셋 품질")
    text("- SWE-Bench를 수정해 SWE-Bench Verified를 만들었다. "), post_link("https://openai.com/index/introducing-swe-bench-verified/")
    text("- 기존 벤치마크의 Platinum 버전을 만든다. "), link("https://arxiv.org/abs/2502.03461")
    image("https://pbs.twimg.com/media/GjICXQlWkAAYnDS?format=jpg&name=4096x4096", width=700)
    image("https://pbs.twimg.com/media/GjICcGQXYAAM4o1?format=jpg&name=4096x4096", width=800)
    text("- 에이전트 벤치마크에는 테스트 부족, 단순한 에이전트도 풀 수 있는 과제 등의 문제가 있다. "), link("https://arxiv.org/abs/2507.02825")
    text("- Docent는 언어 모델로 에이전트 실행 기록을 살펴 문제를 찾는다. "), post_link("https://transluce.org/introducing-docent")


def how_to_think_about_evaluation():
    text("### 평가하는 목적")
    text("어느 평가가 적절한지는 어떤 질문에 답하려는지에 달려 있다.")
    text("1. 사용자나 기업이 고객 서비스 챗봇 같은 용도에 모델 A와 B 중 어느 것을 구매할지 결정하려 한다.")
    text("2. 연구자가 지능 등 모델 자체의 능력을 측정하려 한다.")
    text("3. 기업 활동이나 정책 결정을 위해 모델이 주는 이익과 피해를 이해하려 한다.")
    text("4. 모델 개발자가 개선에 필요한 의견과 측정 결과를 얻으려 한다.")

    text("### 무엇을 평가하는가?")
    text("- 기반 모델 이전에는 표준화한 학습·테스트 분할로 **방법**을 평가했다.")
    text("- 오늘날에는 대체로 **모델이나 시스템**을 평가한다. 사용 가능한 방법은 제한하지 않는다.")

    text("예외도 있다.")
    text("- NanoGPT 속도 경진: 데이터를 고정하고 특정 검증 손실까지 걸리는 계산 시간을 측정한다.")
    image("images/karpathy-nanogpt-speedrun.png", width=600), post_link("https://x.com/karpathy/status/1846790537262571739")

    text("방법을 평가하면 연구자가 새로운 알고리즘을 개발할 유인이 생긴다.")
    text("모델과 시스템을 평가한 결과는 실제 사용자에게 유용하다.")

    text("어느 경우든 비교 조건을 명확히 정해야 한다.")


if __name__ == "__main__":
    main()
