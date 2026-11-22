import math
import torch.nn.functional as F
import timeit
from typing import Iterable
import torch
from torch import nn
from einops import rearrange, einsum, reduce

from edtrace import text, image, link
from lecture_util import article_link
from gpu_util import cuda_if_available, get_max_memory_usage
from facts import h100_flop_per_sec, h100_bytes_per_sec
from references import deepseek_v3_2_2025, adagrad_2011, nemotron_3_super_2026


def main():
    text("공지")
    text("- CS336 Slack에 참여하세요.")
    text("- **Stanford** 이메일 주소로 Modal에 가입하세요.")
    text("- [AI 사용 지침](https://docs.google.com/document/d/1SZAlExB1qAc9izHt54gwunNpjKE6wXb8Y7yA_e-baK8/edit?tab=t.0)을 읽으세요.")
    text("- [클러스터 사용 안내](https://docs.google.com/document/d/1cHE0iKVyXLJ3XpIs2XuXTmZ-HMmPk2hIPeCvy-AydMg/edit?tab=t.otis27tacaef)를 읽으세요.")
    
    text("Marin의 1e23 FLOP 학습이 끝났고, [결과가 예측과 일치했습니다](https://x.com/WilliamBarrHeld/status/2039373983632814318)!")
    image("https://pbs.twimg.com/media/HE1P1HmaUAAjLXF?format=jpg&name=medium", width=800)

    text("지난 강의: 강좌 개요, 토큰화")
    text("오늘 다룰 내용: 자원 소요량 계산(시스템)")

    text("지난 강의의 질문입니다. 연산량과 메모리가 정해져 있을 때 어느 정도로 좋은 모델을 학습할 수 있을까요?")
    text("바꿔 말하면 계산 **효율**을 최대화하는 문제입니다.")
    text("그러려면 주어진 계산에 필요한 연산량과 메모리를 이해해야 합니다.")

    motivating_questions()

    text("이 강의에서 배울 내용:")
    text("- 작동 원리: PyTorch 연산의 의미는 비교적 간단합니다.")
    text("- 접근 방식: 필요한 자원을 계산하는 일을 잊지 마세요.")
    text("- 경험적 판단: 자원이 어디에 얼마나 쓰이는지 감을 익힙니다. 오늘은 특별한 기계학습 기법이 필요하지 않습니다.")

    # 메모리 소요량 계산
    tensors_basics()
    tensors_memory()
    tensors_on_gpus()

    # 연산량 계산
    tensor_einops()
    tensor_operations_flops()

    arithmetic_intensity()

    # 학습에 필요한 메모리와 연산량 계산
    deep_network()
    gradients_basics()
    gradients_flops()
    optimizer()
    train_loop()
    
    # 메모리를 더 줄이는 방법
    gradient_accumulation()
    activation_checkpointing()

    text("요약:")
    text("- 데이터, 파라미터, 기울기, 활성값, 옵티마이저 상태를 모두 텐서로 표현하고 연산합니다.")
    text("- einops를 사용하면 텐서 연산을 축의 이름으로 생각할 수 있습니다.")
    text("- 학습 한 단계의 연산량은 6 × 데이터 수 × 파라미터 수 FLOP입니다.")
    text("- 산술 집약도와 루프라인 분석: 처리 속도를 제한하는 것은 연산일까요, 메모리일까요?")
    text("- 행렬곱은 연산 성능이, 원소별 연산은 메모리 대역폭이 병목입니다.")
    text("- 기울기 누적과 활성값 체크포인팅은 메모리 사용량을 줄여 더 큰 배치를 쓸 수 있게 합니다.")


def motivating_questions():
    text("**질문**: H100 1,024대로 파라미터 700억 개의 모델을 15조 토큰으로 학습하는 데 얼마나 걸릴까요?")
    total_flops = 6 * 70e9 * 15e12  # @inspect total_flops
    h100_flop_per_sec = 1979e12 / 2
    mfu = 0.5
    flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24  # @inspect flops_per_day
    days = total_flops / flops_per_day  # @inspect days

    text("**질문**: H100 8대에서 AdamW로 학습할 수 있는 가장 큰 모델은 어느 정도일까요?")
    h100_bytes = 80e9  # @inspect h100_bytes
    bytes_per_parameter = 2 + 2 + (4 + 4)  # 파라미터(2), 기울기(2), 옵티마이저 상태(4 + 4) @inspect bytes_per_parameter
    num_parameters = (h100_bytes * 8) / bytes_per_parameter  # @inspect num_parameters
    text("이 계산은 배치 크기와 시퀀스 길이에 따라 달라지는 활성값을 제외하므로, 모델 크기의 상한입니다.")

    text("대략적으로 어림한 계산입니다.")
    text("이렇게 간단한 계산만으로도 필요 자원의 규모를 빠르게 파악할 수 있습니다.")


def tensors_basics():
    text("다음 항목은 모두 텐서로 저장합니다.")
    text("- 데이터")
    text("- 파라미터")
    text("- 기울기")
    text("- 옵티마이저 상태")
    text("- 활성값")

    text("예: DeepSeek v3.2 모델의 파라미터 "), link(deepseek_v3_2_2025)
    link(title="Hugging Face의 DeepSeek v3.2 모델", url="https://huggingface.co/deepseek-ai/DeepSeek-V3.2?show_file_info=model.safetensors.index.json")

    text("텐서의 랭크는 축의 개수입니다.")
    x = torch.zeros(4)        # 1차원 텐서(벡터) @inspect x
    x = torch.zeros(4, 8)     # 2차원 텐서(행렬) @inspect x
    x = torch.zeros(4, 8, 2)  # 3차원 텐서 @inspect x

    text("Transformer에서는 4차원 텐서도 사용합니다.")
    B = 32   # 배치 크기
    S = 16   # 시퀀스 길이
    H = 16   # 헤드 수
    D = 64   # 헤드당 은닉 차원
    x = torch.zeros(B, S, H, D)


def tensors_memory():
    text("텐서의 원소는 대개 부동소수점 수입니다.")

    text("## fp32")
    link(title="위키백과", url="https://en.wikipedia.org/wiki/Single-precision_floating-point_format")
    image("images/fp32.png", width=700)
    text("기본 자료형은 fp32로, float32 또는 단정밀도라고도 합니다.")
    text("과학 계산에서는 전통적으로 fp32가 기본이며, 경우에 따라 배정밀도(fp64)를 사용합니다.")
    text("딥러닝에서는 훨씬 낮은 정밀도로도 계산할 수 있습니다.")

    text("텐서가 메모리를 얼마나 사용하는지 살펴봅시다.")
    text("메모리 사용량은 값의 개수와 각 값의 자료형으로 결정됩니다.")
    x = torch.zeros(4, 8)  # @inspect x
    assert x.dtype == torch.float32  # 기본 자료형
    assert x.numel() == 4 * 8
    assert x.element_size() == 4  # float는 4바이트
    assert get_memory_usage(x) == 4 * 8 * 4  # 128바이트

    text("GPT-3 피드포워드 층에 있는 행렬 하나:")
    assert get_memory_usage(torch.empty(12288 * 4, 12288)) == 2304 * 1024 * 1024  # 2.3 GB @stepover

    text("## fp16")
    link(title="위키백과", url="https://en.wikipedia.org/wiki/Half-precision_floating-point_format")
    image("images/fp16.png", width=400)
    text("fp16은 float16 또는 반정밀도라고도 하며, 메모리 사용량을 줄입니다.")
    x = torch.zeros(4, 8, dtype=torch.float16)  # @inspect x
    assert x.element_size() == 2
    text("다만 표현 범위가 좁습니다. 특히 작은 수에서 문제가 됩니다.")
    x = torch.tensor([1e-8], dtype=torch.float16)  # @inspect x
    assert x == 0  # 언더플로 발생!
    text("학습 중에 이런 일이 발생하면 학습이 불안정해질 수 있습니다.")

    text("## bf16")
    link(title="위키백과", url="https://en.wikipedia.org/wiki/Bfloat16_floating-point_format")
    image("images/bf16.png", width=400)
    text("Google Brain은 이 문제를 해결하려고 2018년 brain floating point(bf16)를 개발했습니다.")
    text("bf16은 fp16과 같은 메모리를 쓰면서 fp32와 같은 수준의 표현 범위를 제공합니다.")
    text("대신 표현 간격이 더 넓습니다. 딥러닝에서는 이 차이가 상대적으로 덜 중요합니다.")
    x = torch.tensor([1e-8], dtype=torch.bfloat16)  # @inspect x
    assert x != 0  # 언더플로가 발생하지 않습니다!

    text("## 혼합 정밀도")
    text("학습에는 어떤 영향이 있을까요?")
    text("- fp32로 학습하면 잘 작동하지만 메모리가 많이 필요합니다.")
    text("- fp16은 물론 bf16으로 학습할 때도 수치적으로 불안정해질 위험이 있습니다.")

    text("해결 방법은 혼합 정밀도 학습입니다. "), link("https://arxiv.org/pdf/1710.03740.pdf")
    text("- 파라미터, 활성값, 기울기에는 bf16을 사용합니다.")
    text("- 옵티마이저 상태에는 fp32를 사용합니다.")

    text("PyTorch는 자동 혼합 정밀도(AMP) 라이브러리를 제공합니다. "), link(title="문서", url="https://pytorch.org/docs/stable/amp.html")
    text("안전한 경우에만 bf16으로 변환을 시도합니다. 행렬곱은 변환하지만 exp는 변환하지 않는 식입니다.")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        x = torch.zeros(4, 8)  # @inspect x

    text("## fp8")
    text("기계학습의 연산 수요를 배경으로 2022년에 fp8이 표준화됐습니다. [입문 자료](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)")
    image("https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/_images/fp8_formats.png", width=600)
    text("H100은 두 가지 FP8 형식인 E4M3(범위 [-448, 448])와 E5M2([-57344, 57344])를 지원합니다.")
    text("참고 자료: "), link("https://arxiv.org/pdf/2209.05433.pdf")

    text("## fp4")
    text("2025년에 NVIDIA는 [nvfp4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)를 개발했습니다.")
    text("값 하나에 4비트만 씁니다.")
    text("표현할 수 있는 값: -6, -4, -3, -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6")
    text("블록마다 별도의 스케일 계수를 써서 실제 표현 범위를 넓힙니다. 같은 블록 안에서 이웃한 값들의 스케일을 서로 독립적으로 바꿀 수는 없습니다.")
    text("Nemotron 3 Super는 NVFP4로 학습했습니다. "), link(nemotron_3_super_2026)

    text("이 처리 중 일부는 NVIDIA 라이브러리 내부에서 수행되며 사용자가 직접 제어할 수 없습니다.")


def tensors_on_gpus():
    text("기본적으로 텐서는 CPU 메모리에 저장됩니다.")
    x = torch.zeros(32, 32)
    assert x.device == torch.device("cpu")

    text("GPU에서는 어떨까요?")
    image("images/cpu-gpu.png", width=600)
    device = cuda_if_available()  # @inspect device

    text("GPU의 대규모 병렬 연산을 활용하려면 텐서를 GPU 메모리로 옮겨야 합니다.")
    x = x.to(device)

    text("처음부터 GPU에서 텐서를 만들 수도 있습니다.")
    with torch.device(device):
        x = torch.zeros(32, 32)
        assert x.device == device


def tensor_einops():
    einops_motivation()

    text("Einops는 축에 이름을 붙여 텐서를 조작하는 라이브러리입니다.")
    text("아인슈타인의 합산 표기법(Einstein, 1916)에서 아이디어를 얻었습니다.")
    link(title="Einops 튜토리얼", url="https://einops.rocks/1-einops-basics/")

    einops_einsum()
    einops_reduce()
    einops_rearrange()
    

def einops_motivation():
    text("기존 PyTorch 코드:")
    x = torch.ones(2, 2, 3)      # batch seq hidden  @inspect x
    y = torch.ones(2, 2, 3)      # batch seq hidden  @inspect y
    z = x @ y.transpose(-2, -1)  # batch seq seq  @inspect z
    text("축을 잘못 지정하기 쉽습니다. -2와 -1은 어느 축일까요?")


def einops_einsum():
    text("Einsum은 축의 대응을 명시하면서 행렬곱을 일반화한 연산입니다.")

    x = torch.ones(3, 4)  # seq1 hidden @inspect x
    y = torch.ones(4, 3)  # hidden seq2 @inspect y

    # 기존 방식
    z = x @ y   # seq1 seq2 @inspect z

    # 새 방식(einops)
    z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")  # @inspect z

    text("조금 더 복잡한 예를 봅시다.")  # @clear x y z

    x = torch.ones(2, 3, 4)  # batch seq1 hidden @inspect x
    y = torch.ones(2, 3, 4)  # batch seq2 hidden @inspect y

    # 기존 방식
    z = x @ y.transpose(-2, -1)  # batch seq1 seq2  @inspect z

    # 새 방식(einops)
    z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")  # @inspect z
    text("출력에 이름이 없는 축은 그 축을 따라 합산합니다.")

    # `...`를 사용하면 임의 개수의 축에 대한 브로드캐스팅도 나타낼 수 있습니다.
    z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")  # @inspect z


def einops_reduce():
    text("하나의 텐서에 합, 평균, 최댓값, 최솟값 같은 축약 연산을 적용할 수 있습니다.")
    x = torch.ones(2, 3, 4)  # batch seq hidden @inspect x

    # 기존 방식
    y = x.sum(dim=-1)  # @inspect y

    # 새 방식(einops)
    y = reduce(x, "... hidden -> ...", "sum")  # @inspect y


def einops_rearrange():
    text("하나의 축에 두 축을 합쳐 놓은 경우가 있습니다.")
    text("이때 둘 중 한 축에만 연산을 적용하고 싶을 수 있습니다.")

    x = torch.ones(3, 8)  # seq total_hidden @inspect x
    text("여기서 `total_hidden`은 `heads * hidden1`을 한 축으로 편 것입니다.")
    w = torch.ones(4, 4)  # hidden1 hidden2 @inspect w

    # `total_hidden`을 `heads`와 `hidden1` 두 축으로 나눈다.
    x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)  # @inspect x

    # `w`로 변환한다.
    x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")  # @inspect x

    # `heads`와 `hidden2`를 다시 한 축으로 합친다.
    x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")  # @inspect x


def tensor_operations_flops():
    text("연산 방식을 살펴봤으니 이제 각각의 계산 비용을 따져 봅시다.")

    text("부동소수점 연산(FLOP)은 덧셈(x + y)이나 곱셈(x y) 같은 기본 연산 한 번을 뜻합니다.")

    text("발음이 같아 혼동하기 쉬운 두 약어가 있습니다.")
    text("- FLOPs: 부동소수점 연산 횟수로, 수행한 계산량을 나타냅니다.")
    text("- FLOP/s: 초당 부동소수점 연산 횟수로, FLOPS라고도 씁니다. 하드웨어 속도를 나타내는 단위입니다.")

    text("## 연산 규모 가늠하기")
    text("GPT-3(2020)의 학습에는 3.14e23 FLOP이 들었습니다. "), article_link("https://lambdalabs.com/blog/demystifying-gpt-3")
    text("GPT-4(2023)의 학습에는 2e25 FLOP이 들었을 것으로 추정됩니다. "), article_link("https://patmcguinness.substack.com/p/gpt-4-details-revealed")

    text("H100의 최대 성능은 희소 연산에서 1,979 teraFLOP/s이고, 희소성을 쓰지 않으면 그 절반입니다. "), link(title="사양", url="https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet")
    h100_flop_per_sec = 1979e12 / 2

    text("H100 8대를 2주 동안 사용할 경우:")
    total_flops = 8 * 2 * (60 * 60 * 24 * 7) * h100_flop_per_sec  # @inspect total_flops

    text("## 선형 모델")
    if torch.cuda.is_available():
        B = 16384  # 데이터 수
        D = 32768  # 데이터 하나의 차원
        K = 8192   # 출력 개수
    else:
        B = 1024
        D = 256
        K = 64

    x = torch.ones(B, D, device=cuda_if_available())
    w = torch.randn(D, K, device=cuda_if_available())
    y = x @ w

    text("이 행렬곱에 필요한 FLOP은 얼마일까요?")
    text("(i, j, k) 조합 하나마다 곱셈 x[i][j] * w[j][k] 한 번과 덧셈 한 번을 수행합니다.")
    actual_num_flops = 2 * B * D * K  # @inspect actual_num_flops

    text("이 연산의 실행 시간도 측정할 수 있습니다.")
    actual_time = benchmark(lambda: x @ w)  # @inspect actual_time

    text("이 연산에서 측정한 실제 FLOP/s:") 
    actual_flop_per_sec = actual_num_flops / actual_time  # @inspect actual_flop_per_sec

    text("각 GPU의 사양표에서 이론상 최대 성능을 확인할 수 있습니다.")
    text("- 예: "), link(title="H100 사양", url="https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306")
    text("FLOP/s는 자료형에 따라 크게 달라집니다.")
    promised_flop_per_sec = get_promised_flop_per_sec(x.dtype)  # @inspect promised_flop_per_sec

    text("## 모델 FLOP 활용률(MFU)")

    text("정의: MFU = 실제 FLOP/s ÷ 명시된 최대 FLOP/s [통신과 부가 비용 제외]")
    mfu = actual_flop_per_sec / promised_flop_per_sec if promised_flop_per_sec else None  # @inspect mfu

    text("보통 MFU가 0.5 이상이면 상당히 좋은 편입니다.")

    text("그런데 왜 MFU가 1에 더 가까워지지 않을까요?")
    text("답을 알아보려면 GPU가 계산하는 과정을 더 자세히 살펴봐야 합니다.")


def arithmetic_intensity():
    image("images/compute-memory.png", width=300)
    text("연산을 수행하는 과정:")
    text("1. 메모리에서 가속기로 입력을 보냅니다.")
    text("2. 계산을 수행합니다.")
    text("3. 가속기에서 메모리로 출력을 보냅니다.")

    text("얼마나 걸릴까요?")

    text("두 가지에 달려 있습니다.")
    text("1. 가속기 속도(FLOP/s)")
    text("2. 메모리 대역폭(바이트/s)")
    assert h100_flop_per_sec == 1979e12 / 2  # 희소성을 쓰지 않으면 절반
    assert h100_bytes_per_sec == 3.35e12

    arithmetic_intensity_relu()
    arithmetic_intensity_gelu()
    arithmetic_intensity_dot_product()
    arithmetic_intensity_matrix_vector_product()
    arithmetic_intensity_matmul()

    # 그림으로 살펴봅시다.
    roofline_plots()


def arithmetic_intensity_relu():
    n = 1024 * 1024
    x = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    y = torch.relu(x)

    bytes = (2 * n) + (2 * n)  # x를 읽고 y를 쓴다(bf16은 수 하나에 2바이트).
    flops = n  # 비교 n번

    communication_time = bytes / h100_bytes_per_sec  # @inspect communication_time
    computation_time = flops / h100_flop_per_sec  # @inspect computation_time

    text("통신과 계산을 완전히 겹쳐 수행할 수 있다고 가정합니다.")
    total_time = max(communication_time, computation_time)  # @inspect total_time

    text("무엇이 병목일까요?")
    text("- 메모리 병목: 통신 시간 > 계산 시간")
    text("- 연산 병목: 계산 시간 > 통신 시간")

    text("이 경우 ReLU는 메모리가 병목입니다.")

    text("다른 방식으로 살펴봅시다.")
    text("가속기의 연산량 대 대역폭 비율: 전송한 바이트당 가속기는 얼마나 계산할 수 있을까요?")
    h100_accelerator_intensity = h100_flop_per_sec / h100_bytes_per_sec  # @inspect h100_accelerator_intensity

    text("산술 집약도: 이 작업은 전송한 바이트당 실제로 얼마나 계산할까요?")
    arithmetic_intensity = flops / bytes  # ~1/4 @inspect arithmetic_intensity

    text("무엇이 병목일까요?")
    text("- 메모리 병목: 산술 집약도 < 가속기의 연산량 대 대역폭 비율")
    text("- 연산 병목: 산술 집약도 > 가속기의 연산량 대 대역폭 비율")

    assert arithmetic_intensity < h100_accelerator_intensity

    text("일반적으로 메모리가 병목인 경우가 많습니다.")
    text("산술 집약도를 높일 수 있을까요?")


def arithmetic_intensity_gelu():
    n = 1024 * 1024
    x = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    y = F.gelu(x)  # GELU(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))

    bytes = (2 * n) + (2 * n)  # x를 읽고 y를 쓴다(bf16은 수 하나에 2바이트).
    flops = 20 * n  # tanh는 다항식 등 여러 방법으로 근사할 수 있다.

    arithmetic_intensity = flops / bytes  # @inspect arithmetic_intensity

    h100_accelerator_intensity = h100_flop_per_sec / h100_bytes_per_sec  # @inspect h100_accelerator_intensity
    assert arithmetic_intensity < h100_accelerator_intensity

    text("GeLU는 전송한 바이트당 ReLU보다 많은 연산을 수행하므로 산술 집약도가 더 높습니다.")
    text("그래도 메모리가 병목입니다.")
    text("즉, 각 연산을 따로 실행하는 경우 ReLU가 GeLU보다 빠르지는 않습니다.")


def arithmetic_intensity_dot_product():
    n = 1024 * 1024
    x = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    w = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    y = x @ w

    bytes = (2 * n) + (2 * n) + 2  # x와 w를 읽고 y를 쓴다.
    flops = 2 * n - 1  # 곱셈 n번, 덧셈 n-1번

    arithmetic_intensity = flops / bytes  # ~1/2 @inspect arithmetic_intensity

    h100_accelerator_intensity = h100_flop_per_sec / h100_bytes_per_sec  # @inspect h100_accelerator_intensity
    assert arithmetic_intensity < h100_accelerator_intensity
    text("메모리 병목!")


def arithmetic_intensity_matrix_vector_product():
    n = 1024
    x = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    w = torch.ones(n, n, dtype=torch.bfloat16, device=cuda_if_available())
    y = x @ w

    bytes = (2 * n) + (2 * n * n) + (2 * n)  # x와 w를 읽고 y를 쓴다.
    flops = n * (2 * n - 1)  # 내적 n번

    arithmetic_intensity = flops / bytes  # ~1 @inspect arithmetic_intensity

    h100_accelerator_intensity = h100_flop_per_sec / h100_bytes_per_sec  # @inspect h100_accelerator_intensity
    assert arithmetic_intensity < h100_accelerator_intensity
    text("메모리 병목!")

def arithmetic_intensity_matmul():
    n = 1024
    x = torch.ones(n, n, dtype=torch.bfloat16, device=cuda_if_available())
    w = torch.ones(n, n, dtype=torch.bfloat16, device=cuda_if_available())
    y = x @ w

    bytes = (2 * n * n) + (2 * n * n) + (2 * n * n)  # x와 w를 읽고 y를 쓴다.
    flops = n * n * (2 * n - 1)  # 내적 n²번

    arithmetic_intensity = flops / bytes  # ~n/3 @inspect arithmetic_intensity

    h100_accelerator_intensity = h100_flop_per_sec / h100_bytes_per_sec  # @inspect h100_accelerator_intensity
    assert arithmetic_intensity > h100_accelerator_intensity
    text("드디어 연산이 병목입니다.")

    text("행렬이 충분히 크면 가속기의 연산 장치를 모두 활용하게 되어 연산 성능이 병목입니다.")
    text("Transformer를 학습할 때는 큰 행렬곱을 사용합니다.")
    text("추론 중에는 행렬과 벡터를 곱합니다. 이것이 추론에서 메모리가 병목이 되는 이유입니다.")

    text("참고: 산술 집약도와 가속기의 연산량 대 대역폭 비율은 정밀도(bf16 또는 fp32)에 따라서도 달라집니다.")


def roofline_plots():
    text("루프라인 그래프로 산술 집약도와 성능의 관계를 볼 수 있습니다.")
    image("https://jax-ml.github.io/scaling-book/assets/img/roofline-improved-1400.webp", width=600)
    text("- 가로축의 각 위치는 특정한 산술 집약도를 지닌 연산에 해당합니다.")
    text("- 구간별로 직선인 각 함수는 특정 하드웨어에 해당합니다.")
    text("- 기울기가 바뀌는 지점은 가속기의 연산량 대 대역폭 비율입니다. 여기서 메모리 병목에서 연산 병목으로 바뀝니다.")

    text("이를 다시 MFU와 연결하면 다음과 같습니다.")
    text("MFU = min(1, 산술 집약도 / 가속기의 연산량 대 대역폭 비율)")

    link(title="참고 자료", url="https://jax-ml.github.io/scaling-book/roofline/")


def gradients_basics():
    text("지금까지는 텐서를 만들고 연산을 적용하는 순전파를 수행했습니다.")
    text("이제 역전파로 기울기를 구합니다.")

    text("간단한 선형 모델을 예로 들겠습니다.")
    text("y = 0.5 (x * w - 5)^2")

    text("순전파: 손실 계산")
    x = torch.tensor([1., 2, 3])
    w = torch.tensor([1., 1, 1], requires_grad=True)  # 기울기를 구할 대상
    pred_y = x @ w
    loss = 0.5 * (pred_y - 5).pow(2)

    text("역전파: 기울기 계산")
    loss.backward()
    assert torch.equal(w.grad, torch.tensor([1, 2, 3]))  # @inspect w.grad


def gradients_flops():
    text("기울기를 구하는 데 필요한 FLOP을 세어 봅시다.")

    image("images/deep-network.png", width=800)

    B = 1024  # 데이터 수
    D = 256   # 차원

    text("단순화한 모델인 2층 선형 신경망을 정의합니다.")
    x = torch.ones(B, D, device=cuda_if_available())
    w1 = torch.randn(D, D, device=cuda_if_available(), requires_grad=True)
    w2 = torch.randn(D, D, device=cuda_if_available(), requires_grad=True)

    # 순전파
    h1 = einsum(x, w1, "batch in, in out -> batch out")  # x @ w1
    h2 = einsum(h1, w2, "batch in, in out -> batch out")  # h1 @ w2
    loss = (h2.mean() - 0)**2  # 모든 출력의 회귀 목표값을 임의로 0으로 둔다.

    # 역전파
    h1.retain_grad()  # 디버깅용
    h2.retain_grad()  # 디버깅용
    loss.backward()

    text("## 한 층을 자세히 살펴보기")
    text("두 번째 층(h2 = h1 @ w2)에 집중해 봅시다.")

    text("**순전파**: 순전파에 필요한 FLOP을 떠올려 봅시다. ")
    num_forward_flops = 2 * B * D * D   # @inspect num_forward_flops

    text("**역전파**: 역전파에 필요한 FLOP은 얼마일까요?")

    text("다음 두 기울기를 구해야 합니다.")
    text("- h1.grad = d loss / d h1")
    text("- w2.grad = d loss / d w2")

    h1_grad = einsum(h2.grad, w2, "batch out, in out -> batch in")
    assert torch.allclose(h1.grad, h1_grad)

    w2_grad = einsum(h2.grad, h1, "batch out, batch in -> in out")
    assert torch.allclose(w2.grad, w2_grad)

    num_backward_flops = (2 * B * D * D) + (2 * B * D * D)  # @inspect num_backward_flops

    text("역전파의 연산 비용은 순전파의 두 배입니다.")

    text("## 모든 층으로 확장")
    text("지금까지는 w2만 살펴봤습니다. 신경망의 모든 파라미터에 같은 계산을 적용해야 합니다.")

    text("모두 합치면 다음과 같습니다.")
    text("- 순전파: 2 × 데이터 수 × 파라미터 수 FLOP")
    text("- 역전파: 4 × 데이터 수 × 파라미터 수 FLOP")
    text("- 합계: 6 × 데이터 수 × 파라미터 수 FLOP")

    text("다층 퍼셉트론(MLP)에 대해 구한 값입니다.")
    text("문맥 길이가 짧을 때는 Transformer에도 잘 맞는 근삿값입니다.")


def deep_network():
    image("images/deep-network.png", width=800)
    text("층이 L개이고 입력, 활성값, 출력이 D차원인 심층 신경망을 생각해 봅시다.")

    # 신경망을 정의한다.
    D = 8  # 입력, 활성값, 출력의 차원
    L = 3  # 층 수
    model = DeepNetwork(dim=D, num_layers=L).to(cuda_if_available())

    num_parameters = get_num_parameters(model)  # @inspect num_parameters @stepover
    assert num_parameters == (D * D) * L

    # 모델에 데이터 배치를 입력한다.
    B = 4  # 배치 크기
    x = torch.randn(B, D, device=cuda_if_available())  # @inspect x
    y = model(x)  # @inspect y


class Block(nn.Module):
    "선형 변환에 이어 비선형 함수 ReLU를 적용하는 단순한 블록."
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.weight  # 선형 변환
        x = F.relu(x)        # 활성 함수
        return x


class DeepNetwork(nn.Module):
    "`dim`차원 벡터를 `dim`차원 벡터로 변환한다."
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim) for i in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 모든 층을 차례로 적용한다.
        for layer in self.layers:
            x = layer(x)  # @stepover
        return x


def optimizer():
    text("앞에서 정의한 심층 신경망을 다시 봅시다.")
    B = 2  # 배치 크기
    D = 4  # 입력, 활성값, 출력의 차원
    L = 3  # 층 수
    model = DeepNetwork(dim=D, num_layers=L).to(cuda_if_available())  # @stepover

    text("AdaGrad 옵티마이저를 정의해 봅시다.")
    text("- 모멘텀 = SGD + 기울기의 지수 이동 평균")
    text("- AdaGrad = SGD + 기울기 제곱을 이용한 평균화")
    text("- RMSProp = 기울기 제곱에 지수 이동 평균을 적용한 AdaGrad")
    text("- Adam = RMSProp + 모멘텀")

    text("AdaGrad "), link(adagrad_2011)
    optimizer = AdaGrad(model.parameters(), lr=0.01)  # @stepover
    state = model.state_dict()  # @inspect state

    # 기울기를 계산한다.
    x = torch.randn(B, D, device=cuda_if_available())
    y = torch.tensor([4., 5.], device=cuda_if_available())
    pred_y = model(x).mean()  # @stepover
    loss = F.mse_loss(input=pred_y, target=y)
    loss.backward()

    # 한 단계 갱신한다.
    optimizer.step()
    optimizer_state = {i: dict(p_state) for i, (p, p_state) in enumerate(optimizer.state.items())}  # @inspect optimizer_state

    # 메모리를 해제한다.
    optimizer.zero_grad(set_to_none=True)

    text("## 메모리")

    num_parameters = D * D * L
    parameter_memory = 2 * num_parameters  # (bf16은 2바이트) @inspect parameter_memory
    gradient_memory = 2 * num_parameters  # (bf16은 2바이트) @inspect gradient_memory
    optimizer_state_memory = 4 * num_parameters  # (fp32는 4바이트) @inspect optimizer_state_memory
    activation_memory = 2 * (B * D * L)  # (bf16은 2바이트) @inspect activation_memory
    text("거듭제곱한 값의 평균을 여러 단계에 걸쳐 누적하므로, 안정성을 위해 보통 fp32를 사용합니다.")
    text("옵티마이저 상태에 필요한 메모리:")
    text("- AdaGrad: 2차 모멘트를 저장하므로 파라미터당 4바이트")
    text("- Adam: 1차 및 2차 모멘트를 저장하므로 파라미터당 8바이트")

    # 전체 사용량 합산
    total_memory = parameter_memory + activation_memory + gradient_memory + optimizer_state_memory  # @inspect total_memory

    text("## 연산량(학습 한 단계)")
    num_parameters = D * D * L
    flops = 6 * B * num_parameters  # @inspect flops

    text("## Transformer")
    text("Transformer에 필요한 자원을 계산하는 과정은 더 복잡하지만 기본 원리는 같습니다.")
    text("과제 1에서 직접 계산합니다.")

    text("Transformer 학습의 메모리 사용량을 설명하는 글 "), article_link("https://erees.dev/transformer-memory/")
    text("Transformer의 FLOP을 설명하는 글: "), article_link("https://www.adamcasson.com/posts/transformer-flops")


class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super(AdaGrad, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                # 옵티마이저 상태
                state = self.state[p]
                grad = p.grad.data

                # 기울기 제곱의 누적합 g2 = sum_{i<t} g_i^2를 얻는다.
                g2 = state.get("g2", torch.zeros_like(grad))

                # 옵티마이저 상태를 갱신한다.
                g2 += torch.square(grad)
                state["g2"] = g2

                # 파라미터를 갱신한다.
                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)


def train_loop():
    # 실제 데이터를 생성하는 선형 함수의 가중치는 (0, 1, 2, ..., D-1)이다.
    D = 16  # 차원
    true_w = torch.arange(D, dtype=torch.float32, device=cuda_if_available())

    # (x, y) 쌍을 생성하는 데이터 로더
    B = 4  # 배치 크기
    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(B, D).to(cuda_if_available())
        true_y = x @ true_w
        return (x, true_y)

    # 모델과 옵티마이저를 정의한다.
    L = 2  # 층 수
    model = DeepNetwork(dim=D, num_layers=L).to(cuda_if_available()) # @stepover
    optimizer = AdaGrad(model.parameters(), lr=0.01) # @stepover

    # 학습 시작
    num_train_steps = 3
    for t in range(num_train_steps):
        # 데이터를 가져온다.
        x, y = get_batch()

        # 순전파(손실 계산)
        pred_y = model(x).mean()  # @stepover
        loss = F.mse_loss(pred_y, y)

        # 역전파(기울기 계산)
        loss.backward()

        # 파라미터를 갱신한다.
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def gradient_accumulation():
    text("배치를 크게 하면 학습 안정성이 향상됩니다.")
    text("하지만 활성값 메모리도 배치 크기에 비례하므로 메모리가 부족해질 수 있습니다.")
    B = 64     # 배치 크기
    D = 1024   # 차원
    L = 16     # 층 수
    activation_memory = 2 * B * D * L  # (bf16은 2바이트) @inspect activation_memory
    text("기울기 누적:")
    text("- 마이크로배치별로 기울기를 계산합니다.")
    text("- 기울기를 0으로 초기화하지 않고 누적합니다.")
    text("- batch_size / micro_batch_size 단계마다 파라미터를 갱신하고 기울기를 0으로 초기화합니다.")
    micro_batch_size = B / 4
    activation_memory = 2 * micro_batch_size * D * L  # (bf16은 2바이트) @inspect activation_memory


def activation_checkpointing():
    text("학습할 때는 모든 층의 활성값을 저장해야 합니다.")
    text("추론 때는 기울기를 계산하지 않으므로 현재 층의 활성값만 저장하면 됩니다.")

    image("images/deep-network.png", width=800)
    text("필요한 메모리:")
    B = 64     # 배치 크기
    D = 1024   # 차원
    L = 16     # 층 수

    x = torch.randn(B, D, device=cuda_if_available(), requires_grad=True)
    activation_memory = 2 * B * D * L  # @inspect activation_memory

    model = DeepNetwork(dim=D, num_layers=L).to(cuda_if_available())  # @stepover
    memory = get_max_memory_usage(lambda: model(x).sum().backward())  # @inspect memory @stepover

    text("이 메모리를 줄일 수 있을까요?")

    text("활성값 체크포인팅 = 기울기 체크포인팅 = 재구체화(rematerialization)")
    text("기본 원리:")
    text("- 순전파에서는 일부 층의 활성값만 보관합니다.")
    text("- 역전파에서는 직전 체크포인트부터 저장하지 않은 활성값을 다시 계산합니다.")
    text("메모리를 절약하는 대신 계산을 더 수행합니다.")

    # 모든 활성값 저장:    x g1 h1 g2 h2 g3 h3 g4 h4
    # 활성값 체크포인팅:   x    h1    h2    h3    h4

    # 체크포인팅을 사용하는 모델을 정의한다.
    model = DeepNetworkCheckpointed(dim=D, num_layers=L).to(cuda_if_available())  # @stepover
    checkpointed_memory = get_max_memory_usage(lambda: model(x).sum().backward())  # @inspect checkpointed_memory @stepover

    text("특히 L이 큰 깊은 신경망에서 메모리를 더 줄일 수 있을까요?")

    # 모든 층 저장: | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
    # 아무 층도 저장하지 않음: |                            |
    # 일부 층 저장: |    h3       h6          h9 |

    text("몇 층마다 체크포인트를 둘까요?")
    text("- 각 층의 활성값을 저장하면 활성값 메모리는 O(L)이고 재계산은 없습니다.")
    text("- 활성값을 하나도 저장하지 않으면 활성값 메모리는 O(1)이고 연산량은 O(L²)입니다. 층마다 처음부터 다시 계산합니다.")
    text("- sqrt(L)층마다 저장하면 활성값 메모리는 O(sqrt(L))이고 재계산량은 O(L)입니다.")


class DeepNetworkCheckpointed(nn.Module):
    "DeepNetwork에 활성값 체크포인팅을 적용한 모델."
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim) for i in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 모든 층을 차례로 적용한다.
        for layer in self.layers:
            # 체크포인트의 활성값만 저장하고 나머지는 다시 계산한다.
            x = torch.utils.checkpoint.checkpoint(layer, x)  # @stepover
        return x

############################################################

def get_memory_usage(x: torch.Tensor):
    return x.numel() * x.element_size()


def get_promised_flop_per_sec(dtype: torch.dtype) -> float:
    "`device`에서 `dtype`으로 연산할 때의 최대 FLOP/s를 반환한다."
    if not torch.cuda.is_available():
        # 사용할 수 있는 CUDA 장치가 없어 FLOP/s를 구할 수 없다.
        return 1
    properties = torch.cuda.get_device_properties(cuda_if_available())  # @inspect properties.name

    if "A100" in properties.name:
        # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
        if dtype == torch.float32:
            return 19.5e12
        if dtype in (torch.bfloat16, torch.float16):
            return 312e12
        raise ValueError(f"Unknown dtype: {dtype}")

    if "H100" in properties.name:
        # https://www.nvidia.com/en-us/data-center/h100/
        if dtype == torch.float32:
            return 67.5e12
        if dtype in (torch.bfloat16, torch.float16):
            return 1979e12 / 2  # 1979는 희소 연산 기준이며 밀집 연산 성능은 그 절반이다.
        raise ValueError(f"Unknown dtype: {dtype}")

    if "B200" in properties.name:
        # https://www.primeline-solutions.com/media/categories/server/nach-gpu/nvidia-hgx-h200/nvidia-blackwell-b200-datasheet.pdf
        if dtype == torch.float32:
            return 75e12
        if dtype in (torch.bfloat16, torch.float16):
            return 4.5e15 / 2  # 4.5e15는 희소 연산 기준이며 밀집 연산 성능은 그 절반이다.
        raise ValueError(f"Unknown dtype: {dtype}")

    # 알 수 없는 GPU면 호출한 쪽에서 처리할 수 있도록 None을 반환한다.
    return None


def benchmark(func, num_trials: int = 5) -> float:
    "`func`를 실행하는 데 필요한 시간을 초 단위로 반환한다."

    # 앞서 시작한 CUDA 스레드가 종료될 때까지 기다린다.
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    def run():
        # 연산을 수행한다.
        func()

        # CUDA 스레드가 종료될 때까지 기다린다.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # `num_trials`회 연산 시간을 측정한다.
    total_time = timeit.timeit(run, number=num_trials)

    return total_time / num_trials


def get_num_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


if __name__ == "__main__":
    main()
