from dataclasses import dataclass

from sympy import symbols, oo
from edtrace import text, link, image
from lecture_util import article_link
from references import Reference, gqa_2023, mla_2024, longformer_2020, sparse_transformer_2019, mistral_7b_2023, deepseek_v4_2026

# Transformer 모델의 텐서 모양을 나타낼 기호 정의
B, S, T, D, F, N, K, H, L, V = symbols("B S T D F N K H L V", positive=True)
c = symbols("c", positive=True)  # 극한을 계산하기 위한 상수
memory_bandwidth = symbols("memory_bandwidth", positive=True)

scaling_book_transformers = Reference(title="스케일링 교재의 Transformer 장", url="https://jax-ml.github.io/scaling-book/transformers/")
scaling_book_inference = Reference(title="스케일링 교재의 추론 장", url="https://jax-ml.github.io/scaling-book/inference/")

def main():
    text("## 10강. 추론")
    image("images/inference-schema.png", width=600)

    text("### 추론 작업의 특성")
    landscape()
    review_transformer()
    review_of_arithmetic_intensity()
    arithmetic_intensity_of_inference()
    throughput_and_latency()

    text("### 근사를 이용한 가속(손실 있음)")
    reduce_kv_cache_size()
    quantization()
    model_pruning()

    text("추론 복잡도를 줄이면서 정확도 유지하기")

    text("처음부터 학습하는 방법")
    text("1. 더 빠른 모델 구조를 정의한다.")
    text("2. 더 빠른 모델을 학습한다.")

    text("지식 증류를 이용하는 방법")
    text("1. 더 빠른 모델 구조를 정의한다.")
    text("2. 구조가 다른 원래 모델의 가중치를 이용해 초기화한다.")
    text("3. 증류로 더 빠른 모델의 성능을 회복한다.")

    text("### 근사 결과를 검증하는 방법(손실 없음)")
    speculative_sampling()

    text("### 요청이 변하는 환경에서의 추론")
    text("실제 서비스에서 여러 시퀀스를 배치로 묶기 어려운 이유는 다음과 같다.")
    text("1. 요청 도착 시각이 다르다. 배치가 찰 때까지 기다리면 먼저 온 요청이 늦어진다.")
    text("2. 시스템 프롬프트나 여러 응답을 샘플링하는 경우처럼 시퀀스가 앞부분을 공유한다.")
    text("3. 시퀀스 길이가 달라 패딩을 넣으면 비효율적이다.")

    continuous_batching()
    paged_attention()

    text("### 정리")
    text("- 추론은 서비스, 평가, 강화학습에 쓰인다.")
    text("- 메모리 대역폭에 제한되고 작업이 동적으로 변하는 등 학습과 다른 특성이 있다.")
    text("- 새로운 구조, 양자화, 가지치기와 증류, 추측 샘플링 등을 사용할 수 있다.")
    text("- 시스템 분야의 추측 실행과 페이징도 활용한다.")
    text("- 새로운 구조를 설계해 성능을 개선할 여지가 크다.")


def landscape():
    text("추론이 필요한 경우")
    text("- 실제 서비스: 챗봇, 코드 자동완성, 에이전트, 일괄 데이터 처리.")
    text("- 모델 평가: 예를 들어 지시 수행 능력을 평가한다.")
    text("- 강화학습: 응답을 여러 개 샘플링한 뒤 점수를 매긴다.")

    text("**효율**이 중요한 이유: 학습 비용은 한 번 들지만 추론은 여러 번 반복한다.")
    text("- OpenAI는 하루에 약 8.6조 토큰을 처리한다. "), article_link("https://www.pymnts.com/artificial-intelligence-2/2025/openai-bests-google-in-race-for-consumer-ai-token-consumption/")
    text("- 참고로 DeepSeek V4의 학습량은 32조 토큰이었다. "), link(deepseek_v4_2026)
    
    text("추가로 생각할 점")
    text("- 챗봇에서는 토큰 대부분을 사람이 읽으므로 사람이 병목이다.")
    text("- 에이전트에서는 질문 → 내부 실행 기록 → 사용자용 출력의 과정을 거친다. 생성할 토큰 수에 상한이 없을 수 있다.")
    text("- 생성한 토큰 수는 사용한 연산량에 해당한다.")

    text("추론 서비스를 제공하는 기업들. 제품이나 플랫폼이 있는 기업에는 중요한 업무다.")
    text("- 비공개 모델 서비스: OpenAI, Anthropic, Google 등.")
    text("- 가중치 공개 모델 서비스: Together, Fireworks, Baseten, DeepInfra, Groq, Cerebras 등.")

    text("오픈소스 패키지")
    text("- vLLM: 버클리에서 개발했다. PagedAttention을 도입했고 널리 쓰이는 기본 선택지다. "), link(title="GitHub", url="https://github.com/vllm-project/vllm")
    text("- SGLang: 버클리에서 개발했다. RadixAttention을 도입했고 에이전트 작업에 적합하다. "), link(title="프로젝트", url="https://sgl-project.github.io/")
    text("- TensorRT-LLM: NVIDIA에서 개발했으며 GPU에 최적화되어 있다. "), article_link("https://nvidia.github.io/TensorRT-LLM/overview.html")
    text("- llama.cpp: C++로 작성했다. CPU 추론을 지원하며 로컬에서 실행된다. "), link(title="GitHub", url="https://github.com/ggml-org/llama.cpp")

    text("추론 수요가 크므로 실행 속도를 높이는 것이 중요하다.")

    text("‘빠르다’는 것을 어떤 지표로 측정할까?")
    text("- 첫 토큰까지의 시간(TTFT): 생성이 시작될 때까지 사용자가 기다리는 시간. 대화형 서비스에서 중요하다.")
    text("- 지연 시간(초/토큰): 요청 하나에서 토큰이 얼마나 빨리 나타나는지 측정한다. 대화형 서비스에 해당한다.")
    text("- 처리량(토큰/초): 여러 요청 전체에서 나오는 토큰 수를 측정한다. 일괄 처리에 해당한다.")

    text("효율을 결정하는 요인은?")
    text("- 지도학습: 모든 토큰을 알고 있으므로 시퀀스 전체를 병렬 처리할 수 있다(Transformer의 행렬 곱셈).")
    text("- 추론: 토큰을 차례로 생성하므로 생성 단계들을 병렬 처리할 수 없어 연산 장치를 충분히 활용하기 어렵다.")


def review_transformer():
    link(scaling_book_transformers)
    text("표기법(einops와 유사)")
    text("- 기호는 차원과 각 차원의 크기를 나타낸다. B는 배치, T는 시퀀스, D는 모델 차원, H는 헤드 차원이다.")
    text("- 예: BT<font color=\"red\">D</font> x <font color=\"red\">D</font>H → BTH")
    text("- <font color=\"red\">축약 차원(빨간색)</font>은 두 피연산자에 모두 나타나고 결과에서 사라진다.")
    text("- 일반 차원(검은색)은 한 피연산자에 나타나고 결과에 남는다.")
    text("- 예: <font color=\"blue\">B</font><font color=\"red\">D</font> x <font color=\"blue\">B</font><font color=\"red\">D</font> → B")
    text("- <font color=\"blue\">배치 차원(파란색)</font>은 두 피연산자에 모두 나타나고 결과에도 남는다.")

    image("https://jax-ml.github.io/scaling-book/assets/img/transformer-diagram.png", width=800)
    text("표기상의 가정")
    text("- F = 4 D: MLP는 모델 차원의 4배로 상향 투영한다.")
    text("- D = N H: 모델 차원을 N개 헤드로 나눈다.")
    text("- N = K G: GQA에서 헤드를 K개 그룹으로 나눈다.")
    text("- S = T: 학습 중에는 입력 S개를 조건으로 출력 T개를 예측한다.")


def review_of_arithmetic_intensity():
    text("설정: 행렬 X <font color=\"gray\">(B x D)</font>와 W <font color=\"gray\">(D x F)</font>를 곱한다.")
    text("B는 배치 크기, D는 은닉 차원, F는 MLP의 상향 투영 차원이다.")

    text("행렬 곱셈(X * W)의 FLOP 수와 메모리 읽기·쓰기 양을 계산해 보자.")
    flops = 0
    bytes_transferred = 0

    # 연산 수행
    text("1. HBM에서 X <font color=\"gray\">(B x D)</font>를 읽는다.")
    bytes_transferred += 2*B*D   # bf16은 2바이트다.
    text("2. HBM에서 W <font color=\"gray\">(D x F)</font>를 읽는다.")
    bytes_transferred += 2*D*F
    text("3. Y = X <font color=\"gray\">(B x D)</font> @ W <font color=\"gray\">(D x F)</font>를 계산한다.")
    flops += 2*B*D*F
    text("4. Y <font color=\"gray\">(B x F)</font>를 HBM에 쓴다.")
    bytes_transferred += 2*B*F

    assert flops == 2*B*D*F
    assert bytes_transferred == 2*B*D + 2*D*F + 2*B*F

    text("**산술 집약도**는 전송한 바이트당 수행하는 연산량이다. 높은 값이 유리하다.")
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity

    text("B가 D와 F보다 훨씬 작다고 가정하면 다음처럼 단순화할 수 있다.")
    intensity = intensity.subs(D, c*B).subs(F, c*B).limit(c, oo).simplify()  # @inspect intensity
    assert intensity == B

    text("H100의 가속기 연산 성능 대역폭비")
    flops_per_second = 989e12
    memory_bandwidth = 3.35e12
    accelerator_intensity = flops_per_second / memory_bandwidth  # @inspect accelerator_intensity
    assert round(accelerator_intensity) == 295

    text("연산의 산술 집약도가 가속기의 연산 성능 대역폭비보다 높으면 **연산 제한**이다(유리).")
    text("그보다 낮으면 **메모리 대역폭 제한**이다(불리).")
    text("결론: B > 295일 때만 연산 제한이 된다.")

    text("극단적인 예: B = 1이면 행렬과 벡터를 곱한다.")
    text("- 산술 집약도: 1.")
    text("- 메모리 제한이다. D x F 행렬을 읽고 2 D F FLOP만 계산한다.")
    text("- 추론에서는 이와 비슷한 일이 일어난다.")


def arithmetic_intensity_of_inference():
    link(scaling_book_inference)

    image("https://jax-ml.github.io/scaling-book/assets/img/naive-inference-1400.webp", width=800)
    text("단순한 추론은 토큰을 하나 생성할 때마다 이전 토큰 전체를 Transformer에 넣는다.")
    text("T개 토큰을 생성하려면 O(T^3) FLOP이 든다. 순전파 한 번이 O(T^2)이다.")

    text("서로 같은 접두부의 계산은 상당 부분 재사용할 수 있다.")
    text("방법: **KV 캐시**를 HBM에 저장한다.")
    image("https://jax-ml.github.io/scaling-book/assets/img/cached-inference-1400.webp", width=800)
    text("KV 캐시에는 시퀀스(B), 토큰(S), 층(L), 헤드(K) 각각에 해당하는 H차원 벡터를 저장한다.")

    text("추론의 두 단계")
    text("1. **프리필(prefill)**: 프롬프트를 벡터로 인코딩한다. 학습 때처럼 병렬 처리할 수 있다.")
    text("2. **생성**: 새 응답 토큰을 순서대로 생성한다.")

    text("MLP 층과 어텐션 층 각각에서 FLOP 수와 메모리 입출력량을 계산하자.")
    text("S는 조건으로 주어진 토큰 수, T는 생성할 토큰 수다.")
    text("뒤에서 프리필(T = S)과 생성(T = 1)의 경우를 각각 살펴본다.")

    text("### MLP 층(행렬 곱셈만 고려)")
    flops = 0
    bytes_transferred = 0
    
    # 연산 수행
    text("1. HBM에서 X <font color=\"gray\">(B x T x D)</font>를 읽는다.")
    bytes_transferred += 2*B*T*D
    text("2. HBM에서 Wup <font color=\"gray\">(D x F)</font>, Wgate <font color=\"gray\">(D x F)</font>, Wdown <font color=\"gray\">(F x D)</font>을 읽는다.")
    bytes_transferred += 3 * 2*D*F
    text("3. U = X <font color=\"gray\">(B x T x D)</font> @ Wup <font color=\"gray\">(D x F)</font>를 계산한다.")
    flops += 2*B*T*D*F
    text("4. U <font color=\"gray\">(B x T x F)</font>를 HBM에 쓴다.")
    bytes_transferred += 2*B*T*F
    text("5. G = X <font color=\"gray\">(B x T x D)</font> @ Wgate <font color=\"gray\">(D x F)</font>를 계산한다.")
    flops += 2*B*T*D*F
    text("6. G <font color=\"gray\">(B x T x F)</font>를 HBM에 쓴다.")
    bytes_transferred += 2*B*T*F
    text("7. Y = GeLU(G)*U <font color=\"gray\">(B x T x F)</font> @ Wdown <font color=\"gray\">(F x D)</font>을 계산한다.")
    flops += 2*B*T*D*F
    text("8. Y <font color=\"gray\">(B x T x D)</font>를 HBM에 쓴다.")
    bytes_transferred += 2*B*T*D

    assert flops == 6*B*T*D*F
    assert bytes_transferred == 4*B*T*D + 4*B*T*F + 6*D*F

    # 산술 집약도 계산
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity
    text("B*T가 D와 F보다 훨씬 작다고 가정한다.")
    intensity = intensity.subs(D, c*B*T).subs(F, c*B*T).limit(c, oo).simplify()  # @inspect intensity
    assert intensity == B*T

    text("두 단계를 살펴보자.")
    text("1. 프리필: 배치나 시퀀스를 크게 잡아 `B*T`를 충분히 늘리면 연산 제한으로 만들기 쉽다.")
    text("2. 생성에는 두 가지 문제가 있다.")
    text("- 한 번에 토큰 하나만 생성한다(T = 1).")
    text("- B는 동시 요청 수인데 대화형 서비스에서는 예측하기 어렵다.")

    text("### 어텐션 층(FlashAttention의 행렬 곱셈 위주)")
    text("- S는 이전에 생성한 토큰 수다.")
    text("- T는 다음 토큰의 로짓을 계산할 위치 수다.")
    flops = 0
    bytes_transferred = 0
    
    # 연산 수행
    text("1. HBM에서 Q <font color=\"gray\">(B x T x D)</font>, K <font color=\"gray\">(B x S x D)</font>, V <font color=\"gray\">(B x S x D)</font>를 읽는다.")
    bytes_transferred += 2*B*T*D + 2*B*S*D + 2*B*S*D
    text("2. A = Q <font color=\"gray\">(B x T x D)</font> @ K <font color=\"gray\">(B x S x D)</font>를 계산한다.")
    flops += 2*B*S*T*D
    text("3. Y = softmax(A) <font color=\"gray\">(B x S x T x K x G)</font> @ V <font color=\"gray\">(B x S x K x H)</font>를 계산한다.")
    flops += 2*B*S*T*D
    text("4. Y <font color=\"gray\">(B x T x D)</font>를 HBM에 쓴다.")
    bytes_transferred += 2*B*T*D

    assert flops == 4*B*S*T*D
    assert bytes_transferred == 4*B*S*D + 4*B*T*D

    # 산술 집약도 계산
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity
    assert intensity == S*T / (S + T)

    text("두 단계를 살펴보자.")
    text("1. 프리필: T = S")
    prefill_intensity = intensity.subs(T, S).simplify()  # @inspect prefill_intensity
    assert prefill_intensity == S/2  # 유리하다.
    text("2. 생성: T = 1")
    generate_intensity = intensity.subs(T, 1).simplify()  # @inspect generate_intensity
    assert generate_intensity < 1  # 불리하다.

    text("MLP와 달리 B에 의존하지 않으므로 배치를 늘려도 도움이 되지 않는다.")
    text("왜 그럴까?")
    text("- MLP에서는 모든 시퀀스가 같은 가중치를 사용한다(Wup, Wgate, Wdown은 B와 무관).")
    text("- 어텐션에서는 시퀀스마다 KV 캐시 벡터가 다르다(Q, K, V 모두 B에 의존).")

    text("정리")
    text("- 프리필은 연산 제한, 생성은 메모리 대역폭 제한이다.")
    text("- 프리필 MLP 산술 집약도: `B*S`.")
    text("- 프리필 어텐션 산술 집약도: `S/2`.")
    text("- 생성 시 MLP 산술 집약도: `B`(동시 요청 필요).")
    text("- 생성 시 어텐션 산술 집약도: `<1`(이 방법으로 개선 불가).")


@dataclass(frozen=True)
class TransformerPerformanceStats:
    "Transformer 성능 통계. num_params는 매개변수 수(바이트 단위), memory는 매개변수와 KV 캐시를 합친 메모리 사용량(바이트), latency는 토큰 하나를 생성하는 시간(초/토큰), throughput은 초당 생성하는 토큰 수다. "
    num_params: int
    memory: int
    latency: float
    throughput: float

    def substitute(self, key, value):
        "모든 통계에서 `key`를 `value`로 바꾼다."
        return TransformerPerformanceStats(
            self.num_params.subs(key, value).simplify(),
            self.memory.subs(key, value).simplify(),
            self.latency.subs(key, value).simplify(),
            self.throughput.subs(key, value).simplify(),
        )


def compute_transformer_performance_stats(config) -> TransformerPerformanceStats:  # @inspect config
    "`config` 설정으로 Transformer의 성능 통계를 계산한다."

    # Transformer의 매개변수 수
    num_params = 2*V*D + D*F*3*L + (2*D*N*H + 2*D*K*H)*L

    # 매개변수가 차지하는 메모리
    parameter_size = 2*num_params  # bf16은 2바이트다. 학습에는 더 큰 배수가 필요하다.
    
    # 시퀀스 하나의 KV 캐시 크기(S개 토큰, K개 헤드, H차원 헤드, L개 층)
    kv_cache_size_per_seq = S * (K*H) * L * 2 * 2  # 키와 값으로 2배, bf16의 크기로 2배

    # 전체 메모리 사용량
    memory = B * kv_cache_size_per_seq + parameter_size

    # *지연 시간*은 메모리 입출력에 따라 결정된다. 매 스텝에서 모든 매개변수와 KV 캐시를 읽는다.
    latency = memory / memory_bandwidth

    # *처리량*은 지연 시간의 역수에 동시에 생성하는 토큰 수 B를 곱한 값이다.
    throughput = B / latency

    # 설정 대입
    num_params = num_params.subs(config).simplify()  # @inspect num_params
    memory = memory.subs(config).simplify()  # @inspect memory
    latency = latency.subs(config).simplify()  # @inspect latency
    throughput = throughput.subs(config).simplify()  # @inspect throughput

    return TransformerPerformanceStats(num_params, memory, latency, throughput)


def llama2_13b_config(args={}):
    return {
        S: 1024,   # 시퀀스 길이
        D: 5120,   # 모델 차원
        F: 13824,  # 피드포워드 차원
        N: 40,     # 쿼리 헤드 수
        K: 40,     # 키·값 헤드 수
        H: 128,    # 헤드 차원
        L: 40,     # 층 수
        V: 32000,  # 어휘 크기
        memory_bandwidth: 3.35e12,  # 메모리 대역폭(바이트/초)
        **args
    }


def throughput_and_latency():
    text("추론은 메모리 대역폭에 제한된다는 것을 확인했다.")
    text("요청 하나의 이론상 최대 지연 시간과 처리량을 계산해 보자.")
    text("연산과 통신을 완전히 겹칠 수 있고 부수 비용은 무시한다고 가정한다.")

    text("H100에서 Llama 2 13B의 지연 시간과 처리량을 구체적인 값으로 계산한다.")
    config = llama2_13b_config()
    stats = compute_transformer_performance_stats(config)

    # 배치 크기 1
    b1 = stats.substitute(B, 1)  # @inspect b1 @stepover

    # 배치 크기 64
    b64 = stats.substitute(B, 64)  # @inspect b64 @stepover
    text("결과: 지연 시간은 늘고 처리량은 증가한다.")

    # 배치 크기 256
    b256 = stats.substitute(B, 256)  # @inspect b256 @stepover
    text("결과: 지연 시간이 더 늘고 처리량도 더 증가한다.")
    h100_memory = 80e9  # H100 메모리 용량(바이트)
    assert b256.memory > h100_memory  # 메모리에 들어가지 않는다.
    text("결과: 메모리에 들어가지 않으며 처리량 증가폭도 줄어든다.")

    text("배치 크기를 늘린 결과")
    text("- 읽고 쓸 KV 캐시 크기가 O(B)로 증가해 지연 시간이 늘어난다.")
    text("- 매개변수를 읽는 비용을 여러 요청이 분담하므로 처리량은 증가한다.")

    text("지연 시간과 처리량 사이의 **상충 관계**")
    text("1. 작은 배치에서는 지연 시간이 짧고 처리량이 낮다.")
    text("2. 큰 배치에서는 처리량이 높고 지연 시간이 길다.")

    text("간단한 병렬화: 모델 복사본을 M개 실행하면 지연 시간은 같고 처리량은 M배가 된다.")
    text("더 복잡한 병렬화: 모델과 KV 캐시를 분할한다. "), link(scaling_book_inference)

    text("첫 토큰까지의 시간(TTFT)은 주로 프리필 시간에 달려 있다.")
    text("프리필에서는 작은 배치로 첫 토큰까지의 시간을 줄인다.")
    text("생성에서는 큰 배치로 처리량을 높인다.")


def reduce_kv_cache_size():
    text("추론의 병목은 메모리라는 점을 떠올려 보자.")
    text("KV 캐시 크기를 줄여 보자.")
    text("이때 정확도가 지나치게 떨어지지 않아야 한다.")

    text("### 그룹 쿼리 어텐션(GQA) "), link(gqa_2023)
    image("https://jax-ml.github.io/scaling-book/assets/img/gmqa.png", width=800)
    text("쿼리 헤드는 N개, 키·값 헤드는 K개만 둔다. 키·값 헤드 하나에 쿼리 헤드 N/K개를 연결한다.")
    text("다중 헤드 어텐션(MHA): K=N")
    text("다중 쿼리 어텐션(MQA): K=1")
    text("그룹 쿼리 어텐션(GQA): K는 1과 N 사이다.")

    text("지연 시간과 처리량 개선 결과 "), link(gqa_2023)
    image("images/gqa-speed.png", width=500)

    text("GQA가 지연 시간과 처리량을 개선하는 이유는?")
    text("GQA에서는 KV 캐시가 기존 크기의 K/N로 줄어든다.")
    text("메모리 대역폭 제한이므로 메모리 사용량을 줄이면 빨라진다.")

    # 원래의 Llama 2 13B(MHA)
    config = llama2_13b_config({K: 40, B: 64})  # @stepover
    k40_b64 = compute_transformer_performance_stats(config)  # @inspect k40_b64 @stepover

    # K:N 비율이 1:5인 GQA
    config = llama2_13b_config({K: 8, B: 64})  # K:N 비율이 1:5인 GQA 사용 @stepover
    k8_b64 = compute_transformer_performance_stats(config)  # @inspect k8_b64 @stepover
    text("결과: 지연 시간은 길어지지만 처리량은 개선되고 이제 메모리에 들어간다.")

    # 배치 크기를 더 늘릴 수 있다.
    config = llama2_13b_config({K: 8, B: 256})  # 배치 크기 늘리기 @stepover
    k8_b256 = compute_transformer_performance_stats(config)  # @inspect k8_b256 @stepover
    text("결과: 지연 시간은 길어지지만 처리량은 개선되고 여전히 메모리에 들어간다.")

    text("정확도가 떨어지지 않는지 확인하자. "), link(gqa_2023)
    image("images/gqa-accuracy.png", width=800)

    text("### 다중 헤드 잠재 어텐션(MLA) "), link(mla_2024)
    image("images/mla-schema.png", width=800)
    text("일반적인 어텐션의 KV 캐시는 K = W_K h, V = W_V h로 구성된다(N*H차원).")
    text("MLA는 압축 벡터 c = W_c h(C차원)를 저장하고 필요할 때 K = W_K c, V = W_V c로 상향 투영한다.")
    text("DeepSeek V2는 N*H = 16384를 C = 512로 줄였다.")
    text("한 가지 문제는 MLA와 RoPE가 그대로는 호환되지 않는다는 것이다. RoPE용 64차원을 더해 총 512 + 64 = 576차원이 된다.")
    text("앞에서 본 것처럼 KV 캐시를 줄이면 지연 시간과 처리량이 개선된다.")

    text("이제 정확도를 확인해 보자.")
    text("첫째, MHA는 GQA보다 비싸지만 정확도가 높다[표 8]. "), link(mla_2024)
    image("images/mla-accuracy.png", width=800)
    text("둘째, MLA는 MHA보다 정확도가 조금 더 높으면서 비용은 훨씬 낮다[표 9]. "), link(mla_2024)
    image("images/mla-accuracy2.png", width=800)

    text("### 층간 어텐션(CLA) "), link("https://arxiv.org/abs/2405.12981")
    image("images/cla-diagram.png", width=500)
    text("GQA가 헤드끼리 키·값을 공유하듯이 **층 사이**에서 키·값을 공유한다.")
    text("실험에서 정확도와 KV 캐시 크기(지연 시간 및 처리량) 사이의 파레토 경계가 개선됐다.")
    image("images/cla-results.png", width=700)

    text("### 지역 어텐션(슬라이딩 윈도) "), link(longformer_2020), link(sparse_transformer_2019), link(mistral_7b_2023)
    image("images/longformer-attention.png", width=800)
    text("모델링에 가장 관련성이 높은 인접 문맥만 본다.")
    text("유효 문맥 길이는 층 수에 비례해 증가한다.")
    text("KV 캐시가 시퀀스 길이에 의존하지 않는다.")
    text("문제: 정확도가 떨어질 수 있다.")
    text("방법: 지역 어텐션 층과 전역 어텐션 층을 섞어 쓴다.")

    text("### DeepSeek V4의 어텐션")
    text("- 문맥 길이 100만을 지원한다. "),  link(deepseek_v4_2026)
    image("images/deepseek-v4-attention.png", width=800)
    text("- 압축 희소 어텐션(CSA): 토큰 m개를 하나로 압축한다.")
    text("- DeepSeek 희소 어텐션(DSA): 상위 k개를 선택한다.")
    text("- 고압축 어텐션(HCA): 압축률을 더 높인다.")

    text("정리")
    text("- 목표: 추론의 메모리 병목을 줄이기 위해 정확도를 유지하면서 KV 캐시를 줄인다.")
    text("- KV 캐시의 차원을 줄이는 방법: GQA, MLA, CLA.")
    text("- 일부 층에서 지역 어텐션을 사용해 KV 캐시를 일정 길이로 자른다.")
    text("- 다른 방법: 선형 어텐션과 상태 공간 모델(Mamba 2, GatedDeltaNet), 확산 모델.")


def quantization():
    text("수치의 정밀도를 낮춘다.")
    text("추론은 메모리 대역폭에 제한되므로 메모리를 절약하면 지연 시간과 처리량 지표가 개선된다.")
    text("정확도도 고려해야 한다.")

    # 동작 방식
    x = 5.2342  # @inspect x
    scale = 0.1
    zero_point = 4
    x_quant = round(x / scale) + zero_point  # 양자화  @inspect x_quant
    x_approx = (x_quant - zero_point) * scale  # 역양자화 @inspect x_approx

    image("https://www.datocms-assets.com/104802/1709770809-twitter-post-20.png", width=400), article_link("https://www.baseten.co/blog/fp8-efficient-model-inference-with-8-bit-floating-point-numbers/")
    text("- fp32(4바이트): 학습 시 매개변수와 옵티마이저 상태에 필요하다.")
    text("- bf16(2바이트): 추론의 기본 형식이다.")
    text("- fp8(1바이트): H100의 e4m3 범위는 [-240, 240]이다. 학습에도 사용할 수 있다. "), link("https://arxiv.org/pdf/2310.18313")
    text("- int8(1바이트), [-128, 127]: fp8보다 정확도와 비용이 낮으며 추론에만 사용한다. "), link("https://arxiv.org/pdf/2303.17951")
    text("- int4(0.5바이트), [-8, 7]: 비용이 더 낮고 정확도도 더 낮다. "), link("https://arxiv.org/pdf/2303.17951")

    link(title="양자화 방법", url="https://apxml.com/posts/llm-quantization-techniques-explained")

    text("양자화 인지 학습(QAT)")
    text("- 학습 시 순전파에서 양자화와 역양자화를 적용해 양자화 오차를 모사한다.")
    text("- 장점: 양자화할 때의 영향을 반영해 가중치를 학습한다.")
    text("- 단점: 비용이 큰 대규모 학습이 필요하다.")

    text("학습 후 양자화(PTQ)")
    text("- 학습이 끝난 뒤 수행하므로 훨씬 저렴하다.")
    text("- 표본 데이터로 각 층이나 텐서의 스케일과 영점을 정한다.")
    text("- GPTQ: 헤시안 정보를 이용해 아직 양자화하지 않은 가중치를 갱신함으로써 양자화 오차를 반영한다. "), link("https://arxiv.org/abs/2210.17323")

    text("### 활성값 인지 양자화(AWQ)")
    link("https://arxiv.org/abs/2306.00978")
    text("- 일부 활성값 채널은 값이 크다.")
    text("- 이 채널에 연결된 가중치의 영향이 더 크다.")
    text("- 이런 가중치에 더 높은 정밀도를 할당한다.")
    text("- 활성값을 기준으로 높은 정밀도를 유지할 가중치(0.1~1%)를 고른다.")
    text("- fp16 → int3: 메모리 사용량은 1/4로 줄고 속도는 3.2배 향상됐다.")
    image("images/awq-schema.png", width=800)


def model_pruning():
    text("계산 비용이 큰 모델에서 일부 구성 요소를 제거해 비용을 낮춘다.")
    text("그런 다음 성능을 회복한다.")

    text("NVIDIA 논문 "), link("https://arxiv.org/abs/2407.14679")
    image("images/pruning-kd-loop.png", width=600)
    text("알고리즘")
    text("1. 보정용 소규모 데이터셋(표본 1,024개)으로 중요한 층, 헤드, 은닉 차원을 찾는다.")
    text("2. 중요하지 않은 층을 제거해 더 작은 모델을 만든다.")
    text("3. 원래 모델을 가지치기한 모델로 증류한다.")

    text("결과")
    image("images/pruning-kd.png", width=500)

    # TODO


def speculative_sampling():
    text("추론의 두 단계를 떠올려 보자.")
    text("- 프리필: 주어진 시퀀스의 토큰을 병렬로 인코딩한다(연산 제한). 확률도 얻을 수 있다.")
    text("- 생성: 한 번에 한 토큰씩 생성한다(메모리 제한).")
    text("즉, 생성하는 것보다 주어진 결과를 검사하는 것이 더 빠르다.")

    text("추측 샘플링 "), link("https://arxiv.org/abs/2211.17192"), link("https://arxiv.org/abs/2302.01318")
    text("- 비용이 낮은 **초안 모델** p로 토큰 몇 개(예: 4개)를 미리 예측한다.")
    text("- 목표 모델 q가 토큰을 병렬로 평가하고 조건에 맞는 예측을 채택한다.")
    link(title="추측 샘플링 영상", url="https://storage.googleapis.com/gweb-research2023-media/media/SpeculativeDecoding-1-Illustration.mp4")
    article_link("https://research.google/blog/looking-back-at-speculative-decoding/")

    image("images/speculative-sampling-algorithm.png", width=600)
    text("제안 분포 p와 목표 분포 q를 사용하는 거부 샘플링을 수정한 방식이다.")
    text("수정한 점: 적어도 하나의 후보를 항상 생성한다. 일반적인 거부 샘플링은 채택될 때까지 반복한다.")
    text("목표 모델에서 나온 **정확한 표본**이라는 보장이 있다.")

    text("예를 들어 어휘 원소가 {A, B} 두 개라고 가정하고 확인해 보자.")
    text("- 목표 모델의 확률: [q(A), q(B)].")
    text("- 초안 모델의 확률: [p(A), p(B)].")
    text("- p(A) > q(A)라고 가정한다. 초안 모델은 A를 너무 자주 뽑는다.")
    text("- 따라서 p(B) < q(B)이다. 초안 모델은 B를 덜 뽑는다.")
    text("- 잔차 확률 max(q-p, 0): [0, 1].")
    text("추측 샘플링으로 각 토큰이 나올 확률을 계산한다.")
    text("- P[A를 뽑음] = p(A) * (q(A) / p(A)) + p(B) * 1 * 0 = q(A)")
    text("- P[B를 뽑음] = p(B) * 1 + p(A) * (1 - q(A) / p(A)) * 1 = q(B)")

    image("images/speculative-sampling-results.png", width=600)
    image("images/speculative-sampling-stats.png", width=600)

    text("실제 사용 예")
    text("- 목표 모델의 매개변수가 700억 개일 때 초안 모델은 80억 개다.")
    text("- 목표 모델의 매개변수가 80억 개일 때 초안 모델은 10억 개다.")
    text("- 증류를 이용해 초안 모델을 목표 모델에 최대한 가깝게 만든다.")

    text("초안 모델을 개선하는 방법")
    text("- Medusa: 초안 모델이 여러 토큰을 병렬로 생성한다. "), link("https://arxiv.org/abs/2401.10774")
    text("- EAGLE: 초안 모델이 목표 모델의 상위 수준 특징을 입력받는다. "), link("https://arxiv.org/pdf/2401.15077")
    image("images/medusa-eagle.png", width=600)

    text("정리")
    text("- 목표 모델의 분포에서 정확하게 샘플링한다.")
    text("- 검사가 생성보다 빠르다는 성질을 활용한다.")
    text("- 초안 모델은 학습을 통해 개선할 여지가 많다.")


def continuous_batching():
    link(title="Orca: Transformer 기반 생성 모델을 위한 분산 서빙 시스템", url="https://www.usenix.org/system/files/osdi22-yu.pdf"), link(title="발표", url="https://www.youtube.com/watch?v=Ob9PPLxETYU")

    text("문제")
    text("- 학습 데이터는 배치 크기 x 시퀀스 길이의 직사각형 토큰 배열로 구성할 수 있다.")
    text("- 추론 요청은 도착·종료 시점이 서로 달라 각 행의 길이가 다른 배열이 된다.")
    image("https://images.ctfassets.net/xjan103pcp94/1LJioEsEdQQpDCxYNWirU6/82b9fbfc5b78b10c1d4508b60e72fdcf/cb_02_diagram-static-batching.png", width=600)

    text("방법: 반복 단계별 스케줄링")
    text("- 한 단계씩 디코딩한다.")
    text("- 새로 도착한 요청을 배치에 추가한다. 기존 요청의 생성이 끝날 때까지 기다릴 필요가 없다.")

    text("문제")
    text("- 배치 연산은 모든 시퀀스의 차원이 같을 때만 가능할까?")
    text("- 요청마다 길이는 다를 수 있다.")

    text("방법: 선택적 배치 처리")
    text("- 학습에서 모든 시퀀스 길이가 같으면 B x S x H 텐서를 사용한다.")
    text("- 실제로는 [3, H], [9, H], [5, H]처럼 길이가 다를 수 있다.")
    text("- 어텐션 연산에서는 각 시퀀스를 별도로 처리한다.")
    text("- 어텐션 이외의 연산에서는 시퀀스를 [3 + 9 + 5, H]로 이어 붙인다.")


def paged_attention():
    text("PagedAttention과 vLLM을 소개한 논문 "), link("https://arxiv.org/pdf/2309.06180.pdf")

    text("기존 처리 방식")
    text("- 요청이 들어온다.")
    text("- 프롬프트와 응답의 최대 길이를 고려해 KV 캐시 일부를 미리 할당한다.")
    image("images/paged-attention-fragmentation.png", width=800)
    text("문제: 단편화. 하드디스크에서 생기는 문제와 같다.")
    text("- 실제 생성 토큰 수가 훨씬 적을 수 있어 할당 영역 안의 공간을 낭비한다(내부 단편화).")
    text("- 할당 영역 사이에 사용하지 못하는 공간이 생길 수 있다(외부 단편화).")

    text("방법: PagedAttention. 운영체제의 페이징을 떠올려 보자.")
    text("- 한 시퀀스의 KV 캐시를 서로 붙어 있지 않아도 되는 **블록**들로 나눈다.")
    image("images/paged-attention-blocks.png", width=400)

    text("두 요청이 KV 캐시를 공유하는 예")
    image("images/paged-attention-logical.png", width=800)

    text("서로 다른 시퀀스가 KV 캐시를 공유하는 경우")
    image("images/paged-attention-sharing.png", width=600)
    text("- 같은 시스템 프롬프트를 사용한다.")
    text("- 한 프롬프트에서 여러 응답을 샘플링한다(예: 프로그램 합성).")

    text("방법: 접두부를 공유하고 쓰기가 발생하는 블록만 복사한다(copy-on-write).")
    image("images/paged-attention-parallel.png", width=600)

    text("vLLM의 다른 최적화")
    text("- 블록 읽기와 어텐션을 한 커널로 융합해 커널 실행 오버헤드를 줄인다.")
    text("- 최신 커널을 사용한다(FlashAttention, FlashDecoding).")
    text("- CUDA 그래프로 커널 실행 오버헤드를 피한다.")

    text("정리: 운영체제의 페이징을 활용해 요청이 달라지는 환경에서도 메모리를 효율적으로 사용한다.")


if __name__ == "__main__":
    main()
